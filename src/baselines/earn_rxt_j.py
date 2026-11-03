"""
src/baselines/earn_rxt_j.py
===========================

BP1 - Almazroi & Ayub (2023), IEEE Access, doi:10.1109/ACCESS.2023.3339226.
"Online Payment Fraud Detection Model Using Machine Learning Techniques":
EARN feature extraction + Jaya-optimised ResNeXt-GRU (RXT-J) classification.

Pipeline (re-implemented on the MG-ECFM leakage-free causal features)
--------------------------------------------------------------------
1. **Class re-balancing (SMOTE).**  Synthetic minority samples
   ``x_new = x_i + u (x_nn - x_i)``, ``u ~ U(0, 1)``, ``x_nn`` one of the
   ``k`` nearest fraud neighbours of fraud row ``x_i`` (Chawla et al. 2002),
   applied to the training head only - never to the hold-out tail.
2. **EARN - Ensemble Autoencoder + ResNet feature extractor.**
   * Autoencoder branch ``f_AE: x -> z_AE``, decoder ``g_AE: z_AE -> x_hat``.
   * Residual-network branch ``f_Res`` of pre-activation residual blocks
     ``h <- h + W2 sigma(BN(W1 sigma(BN(h))))``.
   * Ensemble descriptor ``F(x) = [z_AE || z_Res || ||x - x_hat||^2]`` learned
     jointly with ``L_EARN = MSE(x, x_hat) + CE_w(head(z_AE || z_Res), y)``.
3. **RXT - ResNeXt-GRU classifier** on ``F(x)``: ``F`` is projected to a
   ``T x C`` token sequence, passed through ResNeXt blocks
   (split-transform-merge: 1x1 bottleneck -> grouped 3-tap convolution with
   cardinality ``G`` -> 1x1 expansion, identity shortcut), then a GRU reads
   the ``T`` tokens and its final hidden state feeds a linear soft-max head.
   Trained with the **cost-sensitive** weighted cross-entropy
   ``L = -sum_i w_{y_i} log p(y_i | x_i)``, ``w_1 / w_0 = c_FN / c_FP`` x
   inverse class frequency.
4. **J - Jaya fine-tuning.**  The output layer ``theta = (W_out, b_out)`` is
   fine-tuned by the parameter-free Jaya algorithm (Rao 2016) minimising the
   cost-sensitive loss on the chronological hold-out tail::

       theta'_i = theta_i + r1 (theta_best - |theta_i|) - r2 (theta_worst - |theta_i|)

   with element-wise ``r1, r2 ~ U(0, 1)`` and greedy acceptance.  Member 0 of
   the population is the gradient-trained solution, so Jaya can only keep or
   improve the hold-out cost.

Layer widths / epochs are exposed in :class:`EARNRXTJConfig`; defaults are
sized for the PaySim causal feature set (about 30 columns).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch import nn

from .base import (
    SEED,
    BaselineModel,
    EarlyStopper,
    SignedLogStandardScaler,
    check_X,
    chronological_holdout,
    config_from_mapping,
    cost_sensitive_class_weights,
    iterate_minibatches,
    stratified_tail_ok,
)


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class EARNRXTJConfig:
    # SMOTE
    use_smote: bool = True
    smote_ratio: float = 0.2          # target n_fraud / n_legit after oversampling
    smote_k: int = 5
    smote_max_synthetic: int = 200_000
    # EARN
    ae_hidden: int = 64
    ae_latent: int = 16
    resnet_width: int = 64
    resnet_blocks: int = 2
    resnet_embedding: int = 32
    earn_epochs: int = 10
    recon_weight: float = 1.0
    # RXT
    rxt_tokens: int = 8
    rxt_channels: int = 32
    cardinality: int = 8
    bottleneck_width: int = 4
    resnext_blocks: int = 2
    gru_hidden: int = 64
    rxt_epochs: int = 15
    # Optimisation
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    dropout: float = 0.1
    fn_cost: float = 25.0
    fp_cost: float = 1.0
    balance_classes: bool = True
    val_fraction: float = 0.15
    patience: int = 5
    # Jaya
    jaya_population: int = 20
    jaya_iterations: int = 30
    jaya_init_scale: float = 0.05
    # Decision
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.smote_ratio <= 0:
            raise ValueError("smote_ratio must be > 0")
        if self.smote_k < 1:
            raise ValueError("smote_k must be >= 1")
        if self.cardinality < 1 or self.bottleneck_width < 1:
            raise ValueError("cardinality and bottleneck_width must be >= 1")
        if self.jaya_population < 2:
            raise ValueError("Jaya needs a population of at least 2")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must lie in [0, 1)")
        if self.fn_cost <= 0 or self.fp_cost <= 0:
            raise ValueError("misclassification costs must be positive")


# --------------------------------------------------------------------------- #
# SMOTE                                                                       #
# --------------------------------------------------------------------------- #
def smote_oversample(X: np.ndarray, y: np.ndarray, ratio: float, k: int,
                     rng: np.random.Generator, max_synthetic: int) -> Tuple[np.ndarray, np.ndarray]:
    """Chawla et al. (2002) SMOTE for the fraud class (label 1).

    Generates ``ratio * n_legit - n_fraud`` synthetic frauds (capped at
    ``max_synthetic``); returns the original rows followed by the synthetic
    ones.  With fewer than two fraud rows nothing can be interpolated and the
    input is returned unchanged.
    """
    minority = X[y == 1]
    n_min, n_maj = minority.shape[0], int((y == 0).sum())
    n_new = int(min(max(round(ratio * n_maj) - n_min, 0), max_synthetic))
    if n_new == 0 or n_min < 2:
        return X, y
    k_eff = min(k, n_min - 1)
    nn_index = NearestNeighbors(n_neighbors=k_eff + 1).fit(minority)
    neigh = nn_index.kneighbors(minority, return_distance=False)[:, 1:]
    base = rng.integers(0, n_min, size=n_new)
    pick = neigh[base, rng.integers(0, k_eff, size=n_new)]
    gap = rng.random((n_new, 1))
    synthetic = minority[base] + gap * (minority[pick] - minority[base])
    X_out = np.vstack([X, synthetic.astype(X.dtype)])
    y_out = np.concatenate([y, np.ones(n_new, dtype=y.dtype)])
    return X_out, y_out


# --------------------------------------------------------------------------- #
# EARN feature extractor                                                      #
# --------------------------------------------------------------------------- #
class PreActResidualBlock(nn.Module):
    """``h + W2 relu(BN(W1 relu(BN(h))))`` with dropout."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(width)
        self.fc1 = nn.Linear(width, width)
        self.bn2 = nn.BatchNorm1d(width)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(width, width)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.fc1(F.relu(self.bn1(h)))
        z = self.fc2(self.drop(F.relu(self.bn2(z))))
        return h + z


class EARNExtractor(nn.Module):
    """Ensemble Autoencoder + ResNet feature extractor."""

    def __init__(self, d_in: int, cfg: EARNRXTJConfig) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, cfg.ae_hidden), nn.ReLU(),
            nn.Linear(cfg.ae_hidden, cfg.ae_latent), nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(cfg.ae_latent, cfg.ae_hidden), nn.ReLU(),
            nn.Linear(cfg.ae_hidden, d_in),
        )
        self.res_in = nn.Linear(d_in, cfg.resnet_width)
        self.res_blocks = nn.Sequential(
            *[PreActResidualBlock(cfg.resnet_width, cfg.dropout) for _ in range(cfg.resnet_blocks)]
        )
        self.res_out = nn.Sequential(nn.BatchNorm1d(cfg.resnet_width), nn.ReLU(),
                                     nn.Linear(cfg.resnet_width, cfg.resnet_embedding))
        self.out_dim = cfg.ae_latent + cfg.resnet_embedding + 1
        self.aux_head = nn.Linear(cfg.ae_latent + cfg.resnet_embedding, 2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_ae = self.encoder(x)
        x_hat = self.decoder(z_ae)
        recon = ((x - x_hat) ** 2).mean(dim=1, keepdim=True)
        z_res = self.res_out(self.res_blocks(self.res_in(x)))
        logits = self.aux_head(torch.cat([z_ae, z_res], dim=1))
        features = torch.cat([z_ae, z_res, torch.log1p(recon)], dim=1)
        return {"x_hat": x_hat, "recon": recon, "logits": logits, "features": features}


# --------------------------------------------------------------------------- #
# RXT: ResNeXt-GRU classifier                                                 #
# --------------------------------------------------------------------------- #
class ResNeXtBlock1d(nn.Module):
    """Aggregated residual transformation (Xie et al. 2017) on a token axis."""

    def __init__(self, channels: int, cardinality: int, bottleneck_width: int) -> None:
        super().__init__()
        inner = cardinality * bottleneck_width
        self.reduce = nn.Conv1d(channels, inner, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(inner)
        self.grouped = nn.Conv1d(inner, inner, kernel_size=3, padding=1,
                                 groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm1d(inner)
        self.expand = nn.Conv1d(inner, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(channels)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.bn1(self.reduce(h)))
        z = F.relu(self.bn2(self.grouped(z)))
        z = self.bn3(self.expand(z))
        return F.relu(h + z)


class ResNeXtGRU(nn.Module):
    """``F(x) -> tokens (T x C) -> ResNeXt blocks -> GRU -> linear(2)``."""

    def __init__(self, d_in: int, cfg: EARNRXTJConfig) -> None:
        super().__init__()
        self.tokens = cfg.rxt_tokens
        self.channels = cfg.rxt_channels
        self.tokenize = nn.Linear(d_in, cfg.rxt_tokens * cfg.rxt_channels)
        self.blocks = nn.Sequential(*[
            ResNeXtBlock1d(cfg.rxt_channels, cfg.cardinality, cfg.bottleneck_width)
            for _ in range(cfg.resnext_blocks)
        ])
        self.gru = nn.GRU(cfg.rxt_channels, cfg.gru_hidden, batch_first=True)
        self.drop = nn.Dropout(cfg.dropout)
        self.out = nn.Linear(cfg.gru_hidden, 2)

    def penultimate(self, f: torch.Tensor) -> torch.Tensor:
        h = self.tokenize(f).view(-1, self.channels, self.tokens)  # (B, C, T)
        h = self.blocks(h)
        _, h_n = self.gru(h.transpose(1, 2))                         # (1, B, H)
        return self.drop(h_n[-1])

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.out(self.penultimate(f))


# --------------------------------------------------------------------------- #
# Jaya optimiser                                                              #
# --------------------------------------------------------------------------- #
def jaya_minimize(objective, theta0: np.ndarray, population: int, iterations: int,
                  init_scale: float, rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, float]]:
    """Rao (2016) Jaya: minimise ``objective(theta)`` from ``theta0``.

    Returns the best vector found and a history summary.  Member 0 is
    ``theta0`` itself so the result is never worse than the start.
    """
    dim = theta0.size
    scale = init_scale * (np.abs(theta0).mean() + 1e-3)
    pop = theta0[None, :] + scale * rng.standard_normal((population, dim))
    pop[0] = theta0
    fit = np.array([objective(p) for p in pop])
    start = float(fit[0])
    for _ in range(iterations):
        best, worst = pop[np.argmin(fit)].copy(), pop[np.argmax(fit)].copy()
        for i in range(population):
            r1 = rng.random(dim)
            r2 = rng.random(dim)
            cand = pop[i] + r1 * (best - np.abs(pop[i])) - r2 * (worst - np.abs(pop[i]))
            f_cand = objective(cand)
            if f_cand < fit[i]:
                pop[i], fit[i] = cand, f_cand
    i_best = int(np.argmin(fit))
    return pop[i_best].copy(), {"jaya_start_loss": start, "jaya_best_loss": float(fit[i_best])}


# --------------------------------------------------------------------------- #
# Baseline                                                                    #
# --------------------------------------------------------------------------- #
class EARNRXTJ(BaselineModel):
    """BP1: SMOTE + EARN feature extraction + Jaya-tuned ResNeXt-GRU."""

    name = "bp1_earn_rxt_j"
    reference = ("A. A. Almazroi and N. Ayub, 'Online Payment Fraud Detection Model Using "
                 "Machine Learning Techniques', IEEE Access, vol. 11, 2023.")

    def __init__(self, config=None, seed: int = SEED) -> None:
        super().__init__(config_from_mapping(EARNRXTJConfig, config), seed)
        self.threshold_ = float(self.config.threshold)
        self.scaler = SignedLogStandardScaler()
        self.earn: Optional[EARNExtractor] = None
        self.rxt: Optional[ResNeXtGRU] = None

    # ------------------------------------------------------------- training
    def _weighted_ce(self, logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, y, weight=w)

    def _train_earn(self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
                    X_val: Optional[torch.Tensor], y_val: Optional[torch.Tensor]) -> Dict[str, float]:
        cfg = self.config
        opt = torch.optim.Adam(self.earn.parameters(), lr=cfg.learning_rate,
                               weight_decay=cfg.weight_decay)
        stopper = EarlyStopper(cfg.patience)
        rng = self._rng(1)
        last = float("nan")
        for epoch in range(cfg.earn_epochs):
            self.earn.train()
            for idx in iterate_minibatches(X.shape[0], cfg.batch_size, rng):
                if idx.size < 2:
                    continue
                out = self.earn(X[idx])
                loss = cfg.recon_weight * F.mse_loss(out["x_hat"], X[idx]) + \
                    self._weighted_ce(out["logits"], y[idx], w)
                opt.zero_grad()
                loss.backward()
                opt.step()
                last = float(loss.item())
            if X_val is not None:
                self.earn.eval()
                with torch.no_grad():
                    out = self.earn(X_val)
                    v = cfg.recon_weight * F.mse_loss(out["x_hat"], X_val) + \
                        self._weighted_ce(out["logits"], y_val, w)
                if stopper.step(float(v), self.earn, epoch):
                    break
        stopper.restore(self.earn)
        self.earn.eval()
        return {"earn_last_train_loss": last, "earn_best_val_loss": stopper.best}

    def _train_rxt(self, Fx: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
                   F_val: Optional[torch.Tensor], y_val: Optional[torch.Tensor]) -> Dict[str, float]:
        cfg = self.config
        opt = torch.optim.Adam(self.rxt.parameters(), lr=cfg.learning_rate,
                               weight_decay=cfg.weight_decay)
        stopper = EarlyStopper(cfg.patience)
        rng = self._rng(2)
        last = float("nan")
        for epoch in range(cfg.rxt_epochs):
            self.rxt.train()
            for idx in iterate_minibatches(Fx.shape[0], cfg.batch_size, rng):
                if idx.size < 2:
                    continue
                loss = self._weighted_ce(self.rxt(Fx[idx]), y[idx], w)
                opt.zero_grad()
                loss.backward()
                opt.step()
                last = float(loss.item())
            if F_val is not None:
                self.rxt.eval()
                with torch.no_grad():
                    v = self._weighted_ce(self.rxt(F_val), y_val, w)
                if stopper.step(float(v), self.rxt, epoch):
                    break
        stopper.restore(self.rxt)
        self.rxt.eval()
        return {"rxt_last_train_loss": last, "rxt_best_val_loss": stopper.best}

    def _jaya_finetune(self, F_val: torch.Tensor, y_val: torch.Tensor,
                       w: torch.Tensor) -> Dict[str, float]:
        cfg = self.config
        with torch.no_grad():
            h = self.rxt.penultimate(F_val)
        W = self.rxt.out.weight.detach().clone()
        b = self.rxt.out.bias.detach().clone()
        shape_w = W.shape
        theta0 = torch.cat([W.reshape(-1), b]).double().cpu().numpy()

        def objective(theta: np.ndarray) -> float:
            t = torch.as_tensor(theta, dtype=torch.float32, device=self.device)
            W_t = t[:W.numel()].view(shape_w)
            b_t = t[W.numel():]
            with torch.no_grad():
                return float(self._weighted_ce(h @ W_t.T + b_t, y_val, w))

        best, report = jaya_minimize(objective, theta0, cfg.jaya_population,
                                     cfg.jaya_iterations, cfg.jaya_init_scale, self._rng(3))
        t = torch.as_tensor(best, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            self.rxt.out.weight.copy_(t[:W.numel()].view(shape_w))
            self.rxt.out.bias.copy_(t[W.numel():])
        return report

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        cfg = self.config
        head, tail = chronological_holdout(X.shape[0], cfg.val_fraction, min_rows=2)
        self.scaler.fit(X[head])
        Z = self.scaler.transform(X)
        Z_head, y_head = Z[head], y[head]
        if int(y_head.sum()) == 0:          # tail took every fraud: train on all rows
            Z_head, y_head, tail = Z, y, np.arange(0)
        use_val = stratified_tail_ok(y[tail])
        n_before_smote = int(y_head.size)
        if cfg.use_smote:
            Z_head, y_head = smote_oversample(Z_head, y_head, cfg.smote_ratio, cfg.smote_k,
                                              self._rng(0), cfg.smote_max_synthetic)
        w = torch.as_tensor(cost_sensitive_class_weights(
            y_head, cfg.fn_cost, cfg.fp_cost, cfg.balance_classes), device=self.device)

        X_t = self._to_tensor(Z_head)
        y_t = torch.as_tensor(y_head, dtype=torch.long, device=self.device)
        X_v = self._to_tensor(Z[tail]) if use_val else None
        y_v = torch.as_tensor(y[tail], dtype=torch.long, device=self.device) if use_val else None

        d = X.shape[1]
        self.earn = EARNExtractor(d, cfg).to(self.device)
        report = self._train_earn(X_t, y_t, w, X_v, y_v)
        with torch.no_grad():
            F_t = self.earn(X_t)["features"]
            F_v = self.earn(X_v)["features"] if use_val else None
        self.rxt = ResNeXtGRU(self.earn.out_dim, cfg).to(self.device)
        report.update(self._train_rxt(F_t, y_t, w, F_v, y_v))
        if use_val:
            report.update(self._jaya_finetune(F_v, y_v, w))
        else:
            report.update({"jaya_start_loss": float("nan"), "jaya_best_loss": float("nan")})
        report.update({"n_after_smote": int(y_head.size), "n_synthetic": int(y_head.size - n_before_smote),
                       "used_validation_tail": bool(use_val),
                       "class_weights": w.tolist()})
        self.fit_report_.update(report)

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """EARN ensemble descriptor ``F(x)`` (for analysis / ablations)."""
        self._check_fitted()
        Z = self._to_tensor(self.scaler.transform(check_X(X, self.n_features_in_)))
        self.earn.eval()
        return self.earn(Z)["features"].cpu().numpy()

    @torch.no_grad()
    def _predict_fraud_proba(self, X: np.ndarray) -> np.ndarray:
        self.earn.eval()
        self.rxt.eval()
        Z = self._to_tensor(self.scaler.transform(X))
        out = []
        for start in range(0, Z.shape[0], 8192):
            f = self.earn(Z[start:start + 8192])["features"]
            out.append(torch.softmax(self.rxt(f), dim=1)[:, 1])
        return torch.cat(out).double().cpu().numpy()
