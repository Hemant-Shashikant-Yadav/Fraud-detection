"""
src/baselines/argus_conformal.py
================================

BP4 - Nayak & Bushara (2026), IEEE Access: ARGUS - a hybrid
Transformer / state-space model (HybridTransSSM) with Gated Token Mixing,
Bayesian Monte-Carlo-dropout uncertainty and *static* split-conformal risk
control for selective fraud classification.

Architecture
------------
* **Feature tokenizer.**  Each scalar feature ``x_j`` becomes a token
  ``e_j = x_j w_j + b_j in R^D``; a learned ``[CLS]`` token is prepended.
* **HybridTransSSM blocks** (repeated ``n_blocks`` times):
  1. pre-norm multi-head self-attention + GELU feed-forward (Transformer);
  2. **Gated Token Mixing (GTM)** state-space layer: a diagonal linear SSM
     ``s_t = A_bar s_{t-1} + B_bar u_t``, ``y_t = C s_t + D u_t`` with
     ``A = -exp(A_log)``, ``A_bar = exp(Delta A)``, ``B_bar = Delta B``
     (zero-order-hold style discretisation), run as a convolution with
     kernel ``K_t = sum_n C_n B_bar_n A_bar_n^t`` in both token directions
     (tabular features have no intrinsic order) and gated
     ``out = W_o[(y_fwd + y_bwd + D u) * SiLU(W_g x)]``.
* **Head.**  ``LayerNorm([CLS]) -> Dropout -> Linear(2)`` trained with
  cost-sensitive cross-entropy.

Uncertainty and risk control
----------------------------
* **MC dropout** (Gal & Ghahramani 2016): ``T`` stochastic passes with every
  dropout active give ``p_bar = mean_t p_t``, the epistemic spread
  ``std_t p_t`` and the BALD mutual information
  ``I = H[p_bar] - mean_t H[p_t]``.  Passes are seeded so scores are
  reproducible.
* **Static split-conformal risk control** (Angelopoulos et al. 2024) on a
  chronological calibration tail, with the monotone false-negative loss
  ``L_i(t) = 1{p_bar_i < t}`` over the ``n_1`` calibration frauds::

      t_hat = sup{ t : (sum_i L_i(t) + B) / (n_1 + 1) <= alpha },  B = 1

  guaranteeing ``E[FNR] <= alpha`` under exchangeability (which drift
  violates - the motivation for MG-ECFM's *adaptive* CS-ACRC).  Rows whose
  MC-dropout spread exceeds the calibration ``abstain_quantile`` are
  deferred by :meth:`ARGUSConformal.predict_selective`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import (
    SEED,
    BaselineModel,
    EarlyStopper,
    SignedLogStandardScaler,
    check_X,
    config_from_mapping,
    cost_sensitive_class_weights,
    iterate_minibatches,
    stratified_tail_ok,
)


@dataclass
class ARGUSConfig:
    d_model: int = 32
    n_heads: int = 4
    n_blocks: int = 2
    ff_mult: int = 2
    ssm_state: int = 16
    dropout: float = 0.1
    epochs: int = 20
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    fn_cost: float = 25.0
    fp_cost: float = 1.0
    balance_classes: bool = True
    val_fraction: float = 0.1
    cal_fraction: float = 0.2
    patience: int = 5
    mc_samples: int = 30
    crc_alpha: float = 0.1            # target false-negative rate
    abstain_quantile: float = 0.95
    inference_chunk: int = 2048
    fallback_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 < self.crc_alpha < 1.0:
            raise ValueError("crc_alpha must lie in (0, 1)")
        if self.mc_samples < 1:
            raise ValueError("mc_samples must be >= 1")
        if self.val_fraction < 0 or self.cal_fraction <= 0 or \
                self.val_fraction + self.cal_fraction >= 1.0:
            raise ValueError("need val_fraction >= 0, cal_fraction > 0, sum < 1")
        if not 0.0 < self.dropout < 1.0:
            raise ValueError("MC dropout needs 0 < dropout < 1")


# --------------------------------------------------------------------------- #
# Network                                                                     #
# --------------------------------------------------------------------------- #
class FeatureTokenizer(nn.Module):
    """``x (B, d) -> [CLS, x_1 w_1 + b_1, ..., x_d w_d + b_d]`` of shape (B, d+1, D)."""

    def __init__(self, n_features: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x[:, :, None] * self.weight[None] + self.bias[None]
        return torch.cat([self.cls.expand(x.shape[0], -1, -1), tokens], dim=1)


class DiagonalSSMKernel(nn.Module):
    """Convolution kernel ``K_t[c] = sum_n C[c,n] B_bar[c,n] A_bar[c,n]^t``."""

    def __init__(self, channels: int, state: int) -> None:
        super().__init__()
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state + 1, dtype=torch.float32))
                                  .repeat(channels, 1))
        self.B = nn.Parameter(torch.randn(channels, state) / math.sqrt(state))
        self.C = nn.Parameter(torch.randn(channels, state) / math.sqrt(state))
        dt = torch.exp(torch.empty(channels).uniform_(math.log(1e-2), math.log(1e-1)))
        self.log_dt = nn.Parameter(torch.log(dt))

    def forward(self, length: int) -> torch.Tensor:
        dt = torch.exp(self.log_dt)[:, None]                  # (C, 1)
        A = -torch.exp(self.A_log)                            # (C, N), strictly stable
        A_bar = torch.exp(dt * A)                             # in (0, 1)
        B_bar = dt * self.B
        t = torch.arange(length, dtype=A.dtype)
        powers = A_bar[None] ** t[:, None, None]              # (L, C, N)
        return (powers * (self.C * B_bar)[None]).sum(-1)      # (L, C)


def causal_toeplitz_conv(u: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """``y_i = sum_{j <= i} K_{i-j} * u_j`` per channel; u: (B, L, C), K: (L, C)."""
    L = u.shape[1]
    idx = torch.arange(L)
    lag = idx[:, None] - idx[None, :]                         # (L, L)
    T = kernel[lag.clamp(min=0)] * (lag >= 0).to(kernel.dtype)[:, :, None]  # (L, L, C)
    return torch.einsum("ijc,bjc->bic", T, u)


class GatedTokenMixing(nn.Module):
    """Bidirectional diagonal-SSM token mixer with SiLU gating (pre-norm residual)."""

    def __init__(self, d_model: int, state: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.kernel_fwd = DiagonalSSMKernel(d_model, state)
        self.kernel_bwd = DiagonalSSMKernel(d_model, state)
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        L = u.shape[1]
        y_f = causal_toeplitz_conv(u, self.kernel_fwd(L))
        y_b = causal_toeplitz_conv(u.flip(1), self.kernel_bwd(L)).flip(1)
        y = (y_f + y_b + self.D * u) * F.silu(gate)
        return x + self.drop(self.out_proj(y))


class TransformerBlock(nn.Module):
    """Pre-norm MHSA + FFN with explicit dropout modules (MC-dropout friendly)."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_mult * d_model), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(ff_mult * d_model, d_model))
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(a)
        return x + self.drop2(self.ff(self.norm2(x)))


class HybridTransSSM(nn.Module):
    def __init__(self, n_features: int, cfg: ARGUSConfig) -> None:
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, cfg.d_model)
        blocks = []
        for _ in range(cfg.n_blocks):
            blocks.append(TransformerBlock(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout))
            blocks.append(GatedTokenMixing(cfg.d_model, cfg.ssm_state, cfg.dropout))
        self.blocks = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head_drop = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.tokenizer(x))
        return self.head(self.head_drop(self.norm(h[:, 0])))

    def enable_mc_dropout(self) -> None:
        """Eval mode everywhere except dropout layers (Bayesian MC sampling)."""
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


# --------------------------------------------------------------------------- #
# Conformal risk control                                                      #
# --------------------------------------------------------------------------- #
def crc_fnr_threshold(fraud_scores: np.ndarray, alpha: float, B: float = 1.0) -> Dict[str, float]:
    """Largest ``t`` with ``(#{s_i < t} + B) / (n_1 + 1) <= alpha`` (FNR control).

    ``#{s_i < t}`` is non-decreasing in ``t`` so the feasible set is an
    interval ``[0, t_hat]``.  If even ``t = 0`` is infeasible
    (``n_1 + 1 < B / alpha``) the guarantee can only be met by flagging
    everything, so ``t_hat = 0``.
    """
    s = np.sort(np.asarray(fraud_scores, dtype=np.float64))
    n1 = s.size
    budget = alpha * (n1 + 1) - B          # misses allowed
    if n1 == 0 or budget < 0:
        return {"t_hat": 0.0, "feasible": False, "n_calibration_frauds": int(n1),
                "allowed_misses": 0}
    m = int(math.floor(budget + 1e-12))
    t_hat = 1.0 if m >= n1 else float(s[m])  # at most m scores lie strictly below s[m]
    return {"t_hat": t_hat, "feasible": True, "n_calibration_frauds": int(n1),
            "allowed_misses": int(min(m, n1))}


# --------------------------------------------------------------------------- #
# Baseline                                                                    #
# --------------------------------------------------------------------------- #
class ARGUSConformal(BaselineModel):
    """BP4: HybridTransSSM + MC-dropout uncertainty + static split-CRC."""

    name = "bp4_argus_conformal"
    reference = ("Nayak and Bushara, ARGUS: hybrid Transformer-SSM with MC-dropout uncertainty "
                 "and conformal risk control, IEEE Access, 2026.")

    def __init__(self, config=None, seed: int = SEED) -> None:
        super().__init__(config_from_mapping(ARGUSConfig, config), seed)
        self.threshold_ = float(self.config.fallback_threshold)
        self.scaler = SignedLogStandardScaler()
        self.net: Optional[HybridTransSSM] = None
        self.crc_: Dict[str, float] = {}
        self.abstain_std_: float = float("inf")

    # ------------------------------------------------------------- training
    def _split(self, n: int):
        cfg = self.config
        n_cal = max(1, int(round(n * cfg.cal_fraction)))
        n_val = int(round(n * cfg.val_fraction))
        n_train = n - n_cal - n_val
        if n_train < 2:
            raise ValueError("too few rows for ARGUS train / validation / calibration splits")
        return (np.arange(n_train), np.arange(n_train, n_train + n_val),
                np.arange(n_train + n_val, n))

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        cfg = self.config
        tr, va, ca = self._split(X.shape[0])
        if int(y[tr].sum()) == 0:
            raise ValueError("the chronological training head contains no fraud rows")
        self.scaler.fit(X[tr])
        Z = self.scaler.transform(X)
        w = torch.as_tensor(cost_sensitive_class_weights(
            y[tr], cfg.fn_cost, cfg.fp_cost, cfg.balance_classes), device=self.device)
        X_t, y_t = self._to_tensor(Z[tr]), torch.as_tensor(y[tr], dtype=torch.long, device=self.device)
        use_val = stratified_tail_ok(y[va])
        X_v = self._to_tensor(Z[va]) if use_val else None
        y_v = torch.as_tensor(y[va], dtype=torch.long, device=self.device) if use_val else None

        self.net = HybridTransSSM(X.shape[1], cfg).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.learning_rate,
                                weight_decay=cfg.weight_decay)
        stopper = EarlyStopper(cfg.patience)
        rng = self._rng(31)
        last = float("nan")
        for epoch in range(cfg.epochs):
            self.net.train()
            for idx in iterate_minibatches(X_t.shape[0], cfg.batch_size, rng):
                loss = F.cross_entropy(self.net(X_t[idx]), y_t[idx], weight=w)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                last = float(loss.item())
            if use_val:
                self.net.eval()
                with torch.no_grad():
                    v = F.cross_entropy(self.net(X_v), y_v, weight=w)
                if stopper.step(float(v), self.net, epoch):
                    break
        stopper.restore(self.net)
        self.net.eval()

        # static split-conformal risk control on the calibration tail
        mc = self._mc_dropout(Z[ca])
        y_cal = y[ca]
        self.crc_ = crc_fnr_threshold(mc["mean"][y_cal == 1], cfg.crc_alpha)
        if self.crc_["n_calibration_frauds"] > 0:
            self.threshold_ = float(self.crc_["t_hat"])
        self.abstain_std_ = float(np.quantile(mc["std"], cfg.abstain_quantile))
        flagged = mc["mean"] >= self.threshold_
        n1 = max(int(y_cal.sum()), 1)
        self.fit_report_.update({
            "last_train_loss": last,
            "best_val_loss": stopper.best,
            "best_epoch": stopper.best_epoch,
            "crc": dict(self.crc_),
            "crc_threshold": self.threshold_,
            "calibration_fnr": float(((~flagged) & (y_cal == 1)).sum() / n1),
            "calibration_flag_rate": float(flagged.mean()),
            "abstain_std_threshold": self.abstain_std_,
            "n_calibration": int(ca.size),
        })

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def _mc_dropout(self, Z: np.ndarray) -> Dict[str, np.ndarray]:
        """``T`` seeded stochastic passes -> mean, std, entropy, mutual information."""
        cfg = self.config
        T = cfg.mc_samples
        means, stds, ent, mi = [], [], [], []
        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            torch.manual_seed(self.seed + 1009)
            self.net.enable_mc_dropout()
            chunk = max(1, cfg.inference_chunk // T)
            for s in range(0, Z.shape[0], chunk):
                x = self._to_tensor(Z[s:s + chunk])
                rep = x.repeat(T, 1)                                   # (T*b, d)
                p = torch.softmax(self.net(rep), dim=1)[:, 1].view(T, -1).double()
                p_bar = p.mean(0)
                h_bar = _binary_entropy(p_bar)
                means.append(p_bar)
                stds.append(p.std(0, unbiased=False))
                ent.append(h_bar)
                mi.append((h_bar - _binary_entropy(p).mean(0)).clamp(min=0.0))
            self.net.eval()
        return {"mean": torch.cat(means).cpu().numpy(), "std": torch.cat(stds).cpu().numpy(),
                "entropy": torch.cat(ent).cpu().numpy(), "mutual_information": torch.cat(mi).cpu().numpy()}

    def _predict_fraud_proba(self, X: np.ndarray) -> np.ndarray:
        return self._mc_dropout(self.scaler.transform(X))["mean"]

    def predict_with_uncertainty(self, X) -> Dict[str, np.ndarray]:
        """MC-dropout predictive mean plus epistemic diagnostics."""
        self._check_fitted()
        return self._mc_dropout(self.scaler.transform(check_X(X, self.n_features_in_)))

    def predict_selective(self, X) -> np.ndarray:
        """``1`` flag, ``0`` approve, ``-1`` abstain (defer to analyst review)."""
        out = self.predict_with_uncertainty(X)
        decision = (out["mean"] >= self.threshold_).astype(np.int64)
        decision[out["std"] > self.abstain_std_] = -1
        return decision


def _binary_entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-12, 1 - 1e-12)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
