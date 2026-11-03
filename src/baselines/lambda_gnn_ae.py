"""
src/baselines/lambda_gnn_ae.py
==============================

BP2 - Alarfaj & Shahzadi (2025), IEEE Access: graph neural network +
autoencoder fraud detection in a **Lambda architecture**.

Lambda layers
-------------
* **Batch layer (graph representation learning).**  The training
  transactions form a k-NN transaction-similarity graph
  ``G = (V_ref, E)`` in the standardised causal-feature space.  A
  ``L``-layer GraphSAGE network (mean aggregator)::

      h_v^(l) = sigma( W_self^(l) h_v^(l-1) + W_nbr^(l) mean_{u in N(v)} h_u^(l-1) )

  is trained with the cost-sensitive cross-entropy on ``h_v^(L)``.  After
  training, the per-layer *batch views* ``H_ref^(l)`` (l = 0..L-1) of every
  reference node are materialised once.
* **Speed layer (real-time).**  An incoming transaction ``x`` is (a) linked
  to its ``k`` nearest reference nodes and embedded with one message-passing
  step per layer over the stored batch views - no re-training, O(k L) work -
  and (b) scored by an undercomplete autoencoder ``g(f(x))`` trained on
  legitimate traffic only, with reconstruction loss
  ``L(x, g(f(x))) = ||x - g(f(x))||_2^2 / d``.
* **Serving layer.**  The GNN log-odds and the log-normalised reconstruction
  loss are merged by a logistic-regression combiner fitted on the
  chronological hold-out tail (never on rows the GNN was trained on).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch import nn

from .base import (
    SEED,
    BaselineModel,
    EarlyStopper,
    KNNNeighborIndex,
    SignedLogStandardScaler,
    chronological_holdout,
    config_from_mapping,
    cost_sensitive_class_weights,
    iterate_minibatches,
    stratified_tail_ok,
)


@dataclass
class LambdaGNNAEConfig:
    # batch layer (GNN)
    k_neighbors: int = 10
    gnn_hidden: int = 64
    gnn_layers: int = 2
    gnn_epochs: int = 20
    gnn_batch_size: int = 512
    # speed layer (autoencoder)
    ae_hidden: int = 32
    ae_latent: int = 8
    ae_epochs: int = 30
    ae_batch_size: int = 1024
    # optimisation
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    dropout: float = 0.2
    fn_cost: float = 25.0
    fp_cost: float = 1.0
    balance_classes: bool = True
    val_fraction: float = 0.2
    patience: int = 5
    inference_chunk: int = 4096
    # serving layer
    combiner_C: float = 1.0
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.gnn_layers < 1:
            raise ValueError("gnn_layers must be >= 1")
        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must lie in [0, 1)")


# --------------------------------------------------------------------------- #
# Batch layer: GraphSAGE                                                      #
# --------------------------------------------------------------------------- #
class SAGEMeanLayer(nn.Module):
    """``sigma(W_self h_v + W_nbr mean_u h_u)`` (Hamilton et al. 2017)."""

    def __init__(self, d_in: int, d_out: int, dropout: float, activate: bool = True) -> None:
        super().__init__()
        self.lin_self = nn.Linear(d_in, d_out)
        self.lin_nbr = nn.Linear(d_in, d_out, bias=False)
        self.drop = nn.Dropout(dropout)
        self.activate = activate

    def forward(self, h_self: torch.Tensor, h_nbr_mean: torch.Tensor) -> torch.Tensor:
        z = self.lin_self(h_self) + self.lin_nbr(h_nbr_mean)
        return self.drop(F.relu(z)) if self.activate else z


class BatchLayerGNN(nn.Module):
    """Stack of mean-aggregator SAGE layers + soft-max head."""

    def __init__(self, d_in: int, cfg: LambdaGNNAEConfig) -> None:
        super().__init__()
        dims = [d_in] + [cfg.gnn_hidden] * cfg.gnn_layers
        self.layers = nn.ModuleList(
            [SAGEMeanLayer(dims[i], dims[i + 1], cfg.dropout) for i in range(cfg.gnn_layers)]
        )
        self.head = nn.Linear(cfg.gnn_hidden, 2)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def embed_reference(self, nodes: torch.Tensor, level: int, X_ref: torch.Tensor,
                        nbr_table: torch.Tensor) -> torch.Tensor:
        """Differentiable ``h^(level)`` of reference nodes via recursive sampling."""
        if level == 0:
            return X_ref[nodes]
        nb = nbr_table[nodes]                                   # (B, k)
        h_self = self.embed_reference(nodes, level - 1, X_ref, nbr_table)
        h_nb = self.embed_reference(nb.reshape(-1), level - 1, X_ref, nbr_table)
        h_nb = h_nb.view(nb.shape[0], nb.shape[1], -1).mean(dim=1)
        return self.layers[level - 1](h_self, h_nb)

    def embed_new(self, x: torch.Tensor, nb: torch.Tensor,
                  views: List[torch.Tensor]) -> torch.Tensor:
        """Speed-layer ``h^(L)`` of unseen rows from stored batch views."""
        h = x
        for level, layer in enumerate(self.layers):
            h_nb = views[level][nb].mean(dim=1)
            h = layer(h, h_nb)
        return h


# --------------------------------------------------------------------------- #
# Speed layer: autoencoder                                                    #
# --------------------------------------------------------------------------- #
class SpeedLayerAutoencoder(nn.Module):
    """Undercomplete AE ``g(f(x))``; ``L(x, g(f(x)))`` is the anomaly score."""

    def __init__(self, d_in: int, hidden: int, latent: int) -> None:
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        self.g = nn.Sequential(nn.ReLU(), nn.Linear(latent, hidden), nn.ReLU(),
                               nn.Linear(hidden, d_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.g(self.f(x))

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.forward(x)) ** 2).mean(dim=1)


# --------------------------------------------------------------------------- #
# Baseline                                                                    #
# --------------------------------------------------------------------------- #
class LambdaGNNAutoencoder(BaselineModel):
    """BP2: batch-layer GraphSAGE + speed-layer autoencoder + serving combiner."""

    name = "bp2_lambda_gnn_ae"
    reference = ("F. K. Alarfaj and S. Shahzadi, graph neural network and autoencoder "
                 "fraud detection in a Lambda architecture, IEEE Access, 2025.")

    def __init__(self, config=None, seed: int = SEED) -> None:
        super().__init__(config_from_mapping(LambdaGNNAEConfig, config), seed)
        self.threshold_ = float(self.config.threshold)
        self.scaler = SignedLogStandardScaler()
        self.knn = KNNNeighborIndex(self.config.k_neighbors)
        self.gnn: Optional[BatchLayerGNN] = None
        self.ae: Optional[SpeedLayerAutoencoder] = None
        self.combiner: Optional[LogisticRegression] = None
        self.X_ref: Optional[torch.Tensor] = None
        self.nbr_table: Optional[torch.Tensor] = None
        self.batch_views: List[torch.Tensor] = []
        self.ae_scale_: float = 1.0

    # ------------------------------------------------------------ batch layer
    @torch.no_grad()
    def _materialise_views(self) -> List[torch.Tensor]:
        """``[H^(0), ..., H^(L-1)]`` for all reference nodes (chunked, eval mode)."""
        self.gnn.eval()
        views = [self.X_ref]
        chunk = self.config.inference_chunk
        for level in range(1, self.gnn.n_layers):
            prev = views[-1]
            out = []
            for s in range(0, prev.shape[0], chunk):
                nb = self.nbr_table[s:s + chunk]
                out.append(self.gnn.layers[level - 1](prev[s:s + chunk], prev[nb].mean(dim=1)))
            views.append(torch.cat(out))
        return views

    @torch.no_grad()
    def _gnn_logits_new(self, Z: torch.Tensor, nb: torch.Tensor) -> torch.Tensor:
        self.gnn.eval()
        out = []
        chunk = self.config.inference_chunk
        for s in range(0, Z.shape[0], chunk):
            h = self.gnn.embed_new(Z[s:s + chunk], nb[s:s + chunk], self.batch_views)
            out.append(self.gnn.head(h))
        return torch.cat(out)

    def _train_gnn(self, y_ref: np.ndarray, w: torch.Tensor, Z_val: Optional[torch.Tensor],
                   nb_val: Optional[torch.Tensor], y_val: Optional[torch.Tensor]) -> Dict[str, float]:
        cfg = self.config
        y_t = torch.as_tensor(y_ref, dtype=torch.long)
        opt = torch.optim.Adam(self.gnn.parameters(), lr=cfg.learning_rate,
                               weight_decay=cfg.weight_decay)
        stopper = EarlyStopper(cfg.patience)
        rng = self._rng(11)
        last = float("nan")
        L = self.gnn.n_layers
        for epoch in range(cfg.gnn_epochs):
            self.gnn.train()
            for idx in iterate_minibatches(y_ref.size, cfg.gnn_batch_size, rng):
                nodes = torch.as_tensor(idx, dtype=torch.long)
                h = self.gnn.embed_reference(nodes, L, self.X_ref, self.nbr_table)
                loss = F.cross_entropy(self.gnn.head(h), y_t[nodes], weight=w)
                opt.zero_grad()
                loss.backward()
                opt.step()
                last = float(loss.item())
            if Z_val is not None:
                self.batch_views = self._materialise_views()
                v = F.cross_entropy(self._gnn_logits_new(Z_val, nb_val), y_val, weight=w)
                if stopper.step(float(v), self.gnn, epoch):
                    break
        stopper.restore(self.gnn)
        self.gnn.eval()
        self.batch_views = self._materialise_views()
        return {"gnn_last_train_loss": last, "gnn_best_val_loss": stopper.best,
                "gnn_best_epoch": stopper.best_epoch}

    # ------------------------------------------------------------ speed layer
    def _train_ae(self, Z_legit: torch.Tensor) -> Dict[str, float]:
        cfg = self.config
        opt = torch.optim.Adam(self.ae.parameters(), lr=cfg.learning_rate)
        rng = self._rng(12)
        last = float("nan")
        for _ in range(cfg.ae_epochs):
            self.ae.train()
            for idx in iterate_minibatches(Z_legit.shape[0], cfg.ae_batch_size, rng):
                loss = self.ae.reconstruction_loss(Z_legit[idx]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                last = float(loss.item())
        self.ae.eval()
        with torch.no_grad():
            err = self.ae.reconstruction_loss(Z_legit).numpy()
        self.ae_scale_ = float(max(np.median(err), 1e-8))
        return {"ae_last_train_loss": last, "ae_median_legit_error": self.ae_scale_}

    @torch.no_grad()
    def reconstruction_score(self, Z: torch.Tensor) -> np.ndarray:
        """``log(L(x, g(f(x))) / median legit loss)`` - the speed-layer signal."""
        self.ae.eval()
        err = self.ae.reconstruction_loss(Z).double().numpy()
        return np.log((err + 1e-12) / self.ae_scale_)

    # ---------------------------------------------------------- serving layer
    def _components(self, Z: torch.Tensor, nb: torch.Tensor) -> np.ndarray:
        logits = self._gnn_logits_new(Z, nb).double().numpy()
        gnn_log_odds = logits[:, 1] - logits[:, 0]
        return np.column_stack([gnn_log_odds, self.reconstruction_score(Z)])

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        cfg = self.config
        head, tail = chronological_holdout(X.shape[0], cfg.val_fraction, min_rows=3)
        if int(y[head].sum()) == 0 or not stratified_tail_ok(y[tail]):
            head, tail = np.arange(X.shape[0]), np.arange(0)
        use_val = tail.size > 0
        self.scaler.fit(X[head])
        Z = self.scaler.transform(X)
        Z_ref, y_ref = Z[head], y[head]

        # batch-layer graph over the reference (training) transactions
        self.knn.fit(Z_ref)
        self.X_ref = self._to_tensor(Z_ref)
        self.nbr_table = torch.as_tensor(self.knn.train_neighbors(Z_ref), dtype=torch.long)
        w = torch.as_tensor(cost_sensitive_class_weights(
            y_ref, cfg.fn_cost, cfg.fp_cost, cfg.balance_classes))

        Z_val = self._to_tensor(Z[tail]) if use_val else None
        nb_val = torch.as_tensor(self.knn.query(Z[tail]), dtype=torch.long) if use_val else None
        y_val = torch.as_tensor(y[tail], dtype=torch.long) if use_val else None

        self.gnn = BatchLayerGNN(X.shape[1], cfg)
        report = self._train_gnn(y_ref, w, Z_val, nb_val, y_val)

        self.ae = SpeedLayerAutoencoder(X.shape[1], cfg.ae_hidden, cfg.ae_latent)
        report.update(self._train_ae(self.X_ref[torch.as_tensor(y_ref == 0)]))

        # serving-layer combiner on rows the GNN never trained on
        if use_val:
            comb_X, comb_y = self._components(Z_val, nb_val), y[tail]
        else:
            nb_self = torch.as_tensor(self.knn.query(Z_ref), dtype=torch.long)
            comb_X, comb_y = self._components(self.X_ref, nb_self), y_ref
        self.combiner = LogisticRegression(C=cfg.combiner_C, class_weight="balanced",
                                           max_iter=1000, random_state=self.seed)
        self.combiner.fit(comb_X, comb_y)
        report.update({
            "graph_nodes": int(Z_ref.shape[0]),
            "graph_edges": int(self.nbr_table.numel()),
            "k_effective": int(self.knn.k),
            "combiner_fit_on": "holdout_tail" if use_val else "training_rows",
            "combiner_coef_gnn": float(self.combiner.coef_[0, 0]),
            "combiner_coef_ae": float(self.combiner.coef_[0, 1]),
        })
        self.fit_report_.update(report)

    # -------------------------------------------------------------- inference
    def _predict_fraud_proba(self, X: np.ndarray) -> np.ndarray:
        Z_np = self.scaler.transform(X)
        Z = self._to_tensor(Z_np)
        nb = torch.as_tensor(self.knn.query(Z_np), dtype=torch.long)
        return self.combiner.predict_proba(self._components(Z, nb))[:, 1]

    def layer_scores(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Separate batch-layer / speed-layer signals for ablation reporting."""
        from .base import check_X

        self._check_fitted()
        Z_np = self.scaler.transform(check_X(X, self.n_features_in_))
        nb = torch.as_tensor(self.knn.query(Z_np), dtype=torch.long)
        comp = self._components(self._to_tensor(Z_np), nb)
        return {"gnn_log_odds": comp[:, 0], "ae_log_recon": comp[:, 1]}
