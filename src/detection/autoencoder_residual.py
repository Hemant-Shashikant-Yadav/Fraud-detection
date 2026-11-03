"""
src/detection/autoencoder_residual.py
=====================================

Denoising autoencoder (DAE) residual engine for MG-ECFM.

The reconstruction residual of a transaction feature vector is

    r_AE(x_t) = || x_t - g_AE(f_AE(x_t)) ||_2

where ``x_t`` is the leakage-free causal feature vector after the shared
signed-log standardisation (fitted on the training split only), ``f_AE`` the
encoder and ``g_AE`` the decoder. During training inputs are corrupted with
isotropic Gaussian noise ``x~ = x + N(0, sigma^2 I)`` and the network
reconstructs the clean ``x`` (Vincent et al., 2008). By default the DAE is fit
on legitimate transactions only, so fraud manifests as a large residual.

Early stopping monitors the reconstruction MSE on the chronologically last
``val_fraction`` of the training rows (no shuffling across time).
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from src.graph.temporal_memory import SignedLogStandardizer

SEED = 42


@dataclass(frozen=True)
class AutoencoderConfig:
    """Hyper-parameters of :class:`DenoisingAutoencoder` / :class:`AutoencoderResidualEngine`."""

    input_dim: int
    hidden_dims: Tuple[int, ...] = (64, 32)
    latent_dim: int = 16
    activation: str = "relu"
    input_noise_std: float = 0.1
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 2048
    epochs: int = 30
    train_on_legitimate_only: bool = True
    early_stopping_patience: int = 5
    val_fraction: float = 0.1

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.latent_dim <= 0:
            raise ValueError("input_dim and latent_dim must be positive")
        object.__setattr__(self, "hidden_dims", tuple(int(h) for h in self.hidden_dims))
        if any(h <= 0 for h in self.hidden_dims):
            raise ValueError("hidden_dims must be positive")
        if self.activation not in ("relu", "elu", "tanh", "gelu"):
            raise ValueError(f"Unsupported activation '{self.activation}'")
        if self.input_noise_std < 0 or not 0 <= self.dropout < 1:
            raise ValueError("invalid noise / dropout")
        if not 0 <= self.val_fraction < 1:
            raise ValueError("val_fraction must lie in [0, 1)")

    @classmethod
    def from_hyperparams(cls, hyperparams: Mapping, input_dim: int, **overrides) -> "AutoencoderConfig":
        ae = hyperparams.get("autoencoder", {}) if hyperparams else {}
        kwargs = dict(
            input_dim=int(input_dim),
            hidden_dims=tuple(ae.get("hidden_dims", (64, 32))),
            latent_dim=int(ae.get("latent_dim", 16)),
            activation=str(ae.get("activation", "relu")),
            input_noise_std=float(ae.get("input_noise_std", 0.1)),
            dropout=float(ae.get("dropout", 0.1)),
            learning_rate=float(ae.get("learning_rate", 1e-3)),
            weight_decay=float(ae.get("weight_decay", 1e-5)),
            batch_size=int(ae.get("batch_size", 2048)),
            epochs=int(ae.get("epochs", 30)),
            train_on_legitimate_only=bool(ae.get("train_on_legitimate_only", True)),
            early_stopping_patience=int(ae.get("early_stopping_patience", 5)),
        )
        kwargs.update(overrides)
        return cls(**kwargs)


_ACTIVATIONS = {"relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh, "gelu": nn.GELU}


class DenoisingAutoencoder(nn.Module):
    """Symmetric MLP autoencoder ``g(f(x))`` with Gaussian input corruption."""

    def __init__(self, config: AutoencoderConfig, seed: int = SEED) -> None:
        super().__init__()
        self.config = config
        act = _ACTIVATIONS[config.activation]
        with torch.random.fork_rng():
            torch.manual_seed(int(seed))
            enc: List[nn.Module] = []
            prev = config.input_dim
            for h in config.hidden_dims:
                enc += [nn.Linear(prev, h), act(), nn.Dropout(config.dropout)]
                prev = h
            enc.append(nn.Linear(prev, config.latent_dim))
            dec: List[nn.Module] = [act()]
            prev = config.latent_dim
            for h in reversed(config.hidden_dims):
                dec += [nn.Linear(prev, h), act(), nn.Dropout(config.dropout)]
                prev = h
            dec.append(nn.Linear(prev, config.input_dim))
            self.encoder = nn.Sequential(*enc)
            self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``f_AE(x)``."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """``g_AE(z)``."""
        return self.decoder(z)

    def corrupt(self, x: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if self.config.input_noise_std <= 0:
            return x
        noise = torch.randn(x.shape, generator=generator, dtype=x.dtype)
        return x + self.config.input_noise_std * noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        """``r_AE(x) = ||x - g(f(x))||_2`` per row (no corruption)."""
        return torch.linalg.vector_norm(x - self.forward(x), ord=2, dim=1)


class AutoencoderResidualEngine:
    """Scaler + DAE wrapper producing ``r_AE`` for NumPy feature matrices."""

    def __init__(self, config: AutoencoderConfig, seed: int = SEED) -> None:
        self.config = config
        self.seed = int(seed)
        self.scaler = SignedLogStandardizer()
        self.model = DenoisingAutoencoder(config, seed=self.seed)
        self.history: List[Dict[str, float]] = []
        self.best_epoch: Optional[int] = None

    @property
    def is_fitted(self) -> bool:
        return self.scaler.is_fitted and self.best_epoch is not None

    def _tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(self.scaler.transform(x), dtype=np.float32))

    def fit(self, features: np.ndarray, labels: Optional[np.ndarray] = None,
            epochs: Optional[int] = None, verbose: bool = False) -> "AutoencoderResidualEngine":
        """Train the DAE on (legitimate) training rows given in chronological order."""
        cfg = self.config
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != cfg.input_dim:
            raise ValueError(f"features must have shape (n, {cfg.input_dim})")
        if labels is not None and cfg.train_on_legitimate_only:
            mask = np.asarray(labels).astype(int) == 0
            if mask.sum() < 2:
                raise ValueError("Need at least two legitimate rows to fit the autoencoder")
            x = x[mask]
        n = x.shape[0]
        n_val = int(round(n * cfg.val_fraction)) if n >= 20 else 0
        x_train, x_val = x[: n - n_val], x[n - n_val:]
        self.scaler.fit(x_train)
        xt = self._tensor(x_train)
        xv = self._tensor(x_val) if n_val > 0 else None

        gen = torch.Generator().manual_seed(self.seed)
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate,
                               weight_decay=cfg.weight_decay)
        loss_fn = nn.MSELoss()
        best_state = copy.deepcopy(self.model.state_dict())
        best_val = float("inf")
        best_epoch = 0
        patience = 0
        n_epochs = int(epochs if epochs is not None else cfg.epochs)
        self.history = []
        for epoch in range(n_epochs):
            self.model.train()
            perm = torch.randperm(xt.shape[0], generator=gen)
            total = 0.0
            for b0 in range(0, xt.shape[0], cfg.batch_size):
                batch = xt[perm[b0:b0 + cfg.batch_size]]
                recon = self.model(self.model.corrupt(batch, gen))
                loss = loss_fn(recon, batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * batch.shape[0]
            train_loss = total / max(xt.shape[0], 1)
            self.model.eval()
            with torch.no_grad():
                ref = xv if xv is not None else xt
                val_loss = float(loss_fn(self.model(ref), ref))
            self.history.append({"epoch": float(epoch), "train_mse": train_loss, "val_mse": val_loss})
            if verbose:
                print(f"[dae] epoch {epoch}: train={train_loss:.5f} val={val_loss:.5f}")
            if val_loss < best_val - 1e-7:
                best_val, best_epoch, patience = val_loss, epoch, 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                patience += 1
                if patience >= cfg.early_stopping_patience:
                    break
        self.model.load_state_dict(best_state)
        self.model.eval()
        self.best_epoch = best_epoch
        return self

    def residual(self, features: np.ndarray, batch_size: int = 65536) -> np.ndarray:
        """Vector of ``r_AE(x_t)`` for every row of ``features``."""
        if not self.scaler.is_fitted:
            raise RuntimeError("AutoencoderResidualEngine is not fitted")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        self.model.eval()
        out = np.empty(x.shape[0], dtype=np.float64)
        with torch.no_grad():
            for b0 in range(0, x.shape[0], batch_size):
                xb = self._tensor(x[b0:b0 + batch_size])
                out[b0:b0 + batch_size] = self.model.residual(xb).double().numpy()
        return out

    def residual_one(self, features: np.ndarray) -> float:
        """Single-transaction fast path."""
        return float(self.residual(np.asarray(features, dtype=np.float64)[None, :])[0])

    def latency_ms(self, features: np.ndarray, repeats: int = 100) -> float:
        """Median single-row residual latency in milliseconds."""
        row = np.asarray(features, dtype=np.float64).reshape(-1)
        samples = []
        for _ in range(int(repeats)):
            t0 = time.perf_counter()
            self.residual_one(row)
            samples.append((time.perf_counter() - t0) * 1000.0)
        return float(np.median(samples))
