"""
src/baselines/care_gnn.py
=========================

BP3 - Y. Dou, Z. Liu, L. Sun, Y. Deng, H. Peng, P. S. Yu, "Enhancing Graph
Neural Network-based Fraud Detectors against Camouflaged Fraudsters",
ACM CIKM 2020 (CARE-GNN).

Components (paper Sec. 4, notation kept)
---------------------------------------
* **Multi-relation graph** ``G = {V, {E_r}_{r=1..R}}``.  Here each relation is
  an inductive k-NN graph over one *view* of the causal features (by default
  the columns are split into ``R = 3`` contiguous views, e.g. amount/type,
  originator behaviour, beneficiary behaviour for the Phase-1 layout), so a
  new transaction can be attached to every relation at scoring time.
* **Label-aware similarity** (Eq. 2-3): a one-layer MLP per GNN layer predicts
  the label from ``h^(l-1)``; ``D^(l)(v, v') = || sigma(MLP^(l)(h_v)) -
  sigma(MLP^(l)(h_v')) ||_1`` with ``sigma`` the soft-max fraud probability,
  trained with ``L_Simi^(l) = CE(MLP^(l)(h_v), y_v)``.
* **Similarity-aware top-p neighbour selector** (Sec. 4.2): for relation ``r``
  keep the ``ceil(p_r^(l) |N_r(v)|)`` neighbours with the smallest distance.
* **Adaptive filtering threshold via reinforcement learning** (Eq. 4-5): the
  per-relation average neighbour distance ``G(D_r^(l))^(e)`` of epoch ``e``
  yields reward ``f = +1`` if ``G^(e-1) - G^(e) >= 0`` else ``-1``; action
  ``p_r <- p_r + tau`` for ``f = +1`` and ``p_r - tau`` otherwise; the
  relation stops updating once ``|sum of the last 10 rewards| <= 2``.
* **Relation-aware aggregation** (Eq. 6-7): intra-relation
  ``h_{v,r}^(l) = ReLU(AGG_r^(l)({h_{v'}^(l-1)}))`` (mean of the selected
  neighbours, relation-specific weight) and inter-relation
  ``h_v^(l) = ReLU(AGG_all^(l)(h_v^(l-1) || sum_r p_r^(l) h_{v,r}^(l)))``
  (the CARE-Weight variant: relations weighted by their thresholds).
* **Loss** (Eq. 8): ``L_CARE = L_GNN + lambda_1 sum_l L_Simi^(l) +
  lambda_2 ||Theta||^2`` with class-balanced under-sampled mini-batches.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import (
    SEED,
    BaselineModel,
    KNNNeighborIndex,
    SignedLogStandardScaler,
    check_X,
    config_from_mapping,
)


@dataclass
class CAREGNNConfig:
    n_relations: int = 3
    relation_views: Optional[List[List[int]]] = None   # explicit column groups
    k_neighbors: int = 10
    embed_dim: int = 64
    n_layers: int = 1
    epochs: int = 31
    batch_size: int = 1024
    learning_rate: float = 0.01
    lambda_1: float = 2.0            # similarity-loss weight
    lambda_2: float = 1e-3           # L2 regularisation (Adam weight decay)
    dropout: float = 0.0
    p_init: float = 0.5
    rl_step_tau: float = 0.02
    rl_window: int = 10
    rl_stop_abs_sum: int = 2
    undersample_ratio: float = 1.0   # legit : fraud per epoch (1.0 = balanced)
    inference_chunk: int = 4096
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.n_relations < 1:
            raise ValueError("n_relations must be >= 1")
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if not 0.0 < self.p_init <= 1.0:
            raise ValueError("p_init must lie in (0, 1]")
        if self.undersample_ratio <= 0:
            raise ValueError("undersample_ratio must be > 0")


# --------------------------------------------------------------------------- #
# RL threshold controller                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class RelationThreshold:
    """RL controller of one (layer, relation) filtering threshold ``p`` (Eq. 4-5)."""

    p: float
    tau: float
    window: int
    stop_abs_sum: int
    prev_distance: Optional[float] = None
    rewards: List[int] = field(default_factory=list)
    history: List[float] = field(default_factory=list)
    converged: bool = False

    def update(self, avg_distance: float) -> None:
        self.history.append(self.p)
        if self.converged or not math.isfinite(avg_distance):
            return
        if self.prev_distance is not None:
            reward = 1 if self.prev_distance - avg_distance >= 0 else -1
            self.rewards.append(reward)
            self.p = float(np.clip(self.p + self.tau * reward, self.tau, 1.0))
            if len(self.rewards) >= self.window and \
                    abs(sum(self.rewards[-self.window:])) <= self.stop_abs_sum:
                self.converged = True
        self.prev_distance = avg_distance


# --------------------------------------------------------------------------- #
# Network                                                                     #
# --------------------------------------------------------------------------- #
class CARELayer(nn.Module):
    """One CARE-GNN layer: similarity MLP, top-p selector, two-level aggregation."""

    def __init__(self, d_in: int, d_out: int, n_relations: int, dropout: float) -> None:
        super().__init__()
        self.sim_mlp = nn.Linear(d_in, 2)                     # label-aware scorer
        self.intra = nn.ModuleList([nn.Linear(d_in, d_out) for _ in range(n_relations)])
        self.inter = nn.Linear(d_in + d_out, d_out)
        self.drop = nn.Dropout(dropout)

    def fraud_score(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.sim_mlp(h)
        return logits, torch.softmax(logits, dim=1)[:, 1]

    def forward(self, h_c: torch.Tensor, h_nbs: Sequence[torch.Tensor],
                p: Sequence[float], s_nbs: Optional[Sequence[torch.Tensor]] = None):
        """``h_c``: (B, d); ``h_nbs[r]``: (B, k, d); returns (h, logits_c, distances)."""
        logits_c, s_c = self.fraud_score(h_c)
        combined = torch.zeros(h_c.shape[0], self.inter.out_features, dtype=h_c.dtype)
        distances: List[float] = []
        for r, h_nb in enumerate(h_nbs):
            B, k, d = h_nb.shape
            if s_nbs is None:
                s_nb = self.fraud_score(h_nb.reshape(-1, d))[1].view(B, k)
            else:
                s_nb = s_nbs[r]
            dist = (s_c.detach()[:, None] - s_nb.detach()).abs()          # L1, Eq. 3
            n_keep = max(1, int(math.ceil(p[r] * k)))
            order = torch.argsort(dist, dim=1)[:, :n_keep]                 # top-p
            kept = torch.gather(h_nb, 1, order[:, :, None].expand(-1, -1, d))
            distances.append(float(torch.gather(dist, 1, order).mean()))
            h_r = F.relu(self.intra[r](kept.mean(dim=1)))                 # intra-rel.
            combined = combined + p[r] * h_r                               # CARE-Weight
        h = F.relu(self.inter(torch.cat([h_c, combined], dim=1)))         # inter-rel.
        return self.drop(h), logits_c, distances


class CARENetwork(nn.Module):
    def __init__(self, d_in: int, cfg: CAREGNNConfig, n_relations: int) -> None:
        super().__init__()
        dims = [d_in] + [cfg.embed_dim] * cfg.n_layers
        self.layers = nn.ModuleList(
            [CARELayer(dims[i], dims[i + 1], n_relations, cfg.dropout) for i in range(cfg.n_layers)]
        )
        self.classifier = nn.Linear(cfg.embed_dim, 2)


# --------------------------------------------------------------------------- #
# Baseline                                                                    #
# --------------------------------------------------------------------------- #
class CAREGNN(BaselineModel):
    """BP3: camouflage-resistant multi-relation GNN (CARE-GNN)."""

    name = "bp3_care_gnn"
    reference = ("Y. Dou et al., 'Enhancing Graph Neural Network-based Fraud Detectors "
                 "against Camouflaged Fraudsters', ACM CIKM 2020.")

    def __init__(self, config=None, seed: int = SEED) -> None:
        super().__init__(config_from_mapping(CAREGNNConfig, config), seed)
        self.threshold_ = float(self.config.threshold)
        self.scaler = SignedLogStandardScaler()
        self.views_: List[np.ndarray] = []
        self.indices: List[KNNNeighborIndex] = []
        self.tables: List[torch.Tensor] = []
        self.net: Optional[CARENetwork] = None
        self.thresholds: List[List[RelationThreshold]] = []
        self.X_ref: Optional[torch.Tensor] = None
        self.ref_views: List[torch.Tensor] = []
        self.ref_scores: List[torch.Tensor] = []

    # ---------------------------------------------------------- relations
    def _resolve_views(self, d: int) -> List[np.ndarray]:
        cfg = self.config
        if cfg.relation_views is not None:
            views = [np.asarray(v, dtype=np.int64) for v in cfg.relation_views]
            for v in views:
                if v.size == 0 or v.min() < 0 or v.max() >= d:
                    raise ValueError(f"relation view {v.tolist()} out of range for d={d}")
            return views
        n_rel = min(cfg.n_relations, d)
        return [np.asarray(v, dtype=np.int64) for v in np.array_split(np.arange(d), n_rel)]

    def _p(self, level: int) -> List[float]:
        return [t.p for t in self.thresholds[level]]

    @property
    def relation_thresholds(self) -> List[List[float]]:
        """Current ``p_r^(l)`` for every layer ``l`` and relation ``r``."""
        return [self._p(level) for level in range(len(self.thresholds))]

    # ------------------------------------------------------------ training
    def _embed(self, nodes: torch.Tensor, level: int,
               collect: Optional[List[Tuple[int, torch.Tensor, List[float]]]] = None) -> torch.Tensor:
        """Differentiable ``h^(level)`` for reference ``nodes`` (recursive sampling)."""
        if level == 0:
            return self.X_ref[nodes]
        h_c = self._embed(nodes, level - 1, collect)
        h_nbs = []
        for table in self.tables:
            nb = table[nodes]
            h_nb = self._embed(nb.reshape(-1), level - 1)
            h_nbs.append(h_nb.view(nb.shape[0], nb.shape[1], -1))
        h, logits_c, dist = self.net.layers[level - 1](h_c, h_nbs, self._p(level - 1))
        if collect is not None:
            collect.append((level - 1, logits_c, dist))
        return h

    def _epoch_nodes(self, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        fraud = np.flatnonzero(y == 1)
        legit = np.flatnonzero(y == 0)
        n_legit = min(legit.size, max(1, int(round(self.config.undersample_ratio * fraud.size))))
        picked = np.concatenate([fraud, rng.choice(legit, size=n_legit, replace=False)])
        return rng.permutation(picked)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        cfg = self.config
        self.scaler.fit(X)
        Z = self.scaler.transform(X)
        self.X_ref = self._to_tensor(Z)
        self.views_ = self._resolve_views(X.shape[1])
        self.indices, self.tables = [], []
        for view in self.views_:
            index = KNNNeighborIndex(cfg.k_neighbors).fit(Z[:, view])
            self.indices.append(index)
            self.tables.append(torch.as_tensor(index.train_neighbors(Z[:, view]), dtype=torch.long))
        R = len(self.views_)
        self.net = CARENetwork(X.shape[1], cfg, R)
        self.thresholds = [[RelationThreshold(cfg.p_init, cfg.rl_step_tau, cfg.rl_window,
                                              cfg.rl_stop_abs_sum) for _ in range(R)]
                           for _ in range(cfg.n_layers)]
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate,
                               weight_decay=cfg.lambda_2)
        y_t = torch.as_tensor(y, dtype=torch.long)
        rng = self._rng(21)
        losses: List[float] = []
        for _ in range(cfg.epochs):
            self.net.train()
            epoch_nodes = self._epoch_nodes(y, rng)
            dist_sum = np.zeros((cfg.n_layers, R))
            dist_cnt = np.zeros((cfg.n_layers, R))
            epoch_loss = 0.0
            for start in range(0, epoch_nodes.size, cfg.batch_size):
                nodes = torch.as_tensor(epoch_nodes[start:start + cfg.batch_size], dtype=torch.long)
                collect: List[Tuple[int, torch.Tensor, List[float]]] = []
                h = self._embed(nodes, cfg.n_layers, collect)
                loss_gnn = F.cross_entropy(self.net.classifier(h), y_t[nodes])
                loss_simi = sum(F.cross_entropy(lg, y_t[nodes]) for _, lg, _ in collect)
                loss = loss_gnn + cfg.lambda_1 * loss_simi
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += float(loss.item()) * nodes.numel()
                for level, _, dist in collect:
                    dist_sum[level] += np.asarray(dist) * nodes.numel()
                    dist_cnt[level] += nodes.numel()
            losses.append(epoch_loss / max(epoch_nodes.size, 1))
            avg = dist_sum / np.maximum(dist_cnt, 1)
            for level in range(cfg.n_layers):
                for r in range(R):
                    self.thresholds[level][r].update(float(avg[level, r]))
        self.net.eval()
        self._materialise_reference()
        self.fit_report_.update({
            "final_epoch_loss": losses[-1] if losses else float("nan"),
            "relation_views": [v.tolist() for v in self.views_],
            "relation_thresholds": self.relation_thresholds,
            "rl_converged": [[t.converged for t in row] for row in self.thresholds],
            "k_effective": [ix.k for ix in self.indices],
        })

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def _materialise_reference(self) -> None:
        """Cache ``H_ref^(l)`` and the similarity scores used by each layer."""
        self.net.eval()
        chunk = self.config.inference_chunk
        views = [self.X_ref]
        scores = []
        for level, layer in enumerate(self.net.layers):
            prev = views[-1]
            scores.append(torch.cat([layer.fraud_score(prev[s:s + chunk])[1]
                                     for s in range(0, prev.shape[0], chunk)]))
            if level == len(self.net.layers) - 1:
                break
            out = []
            for s in range(0, prev.shape[0], chunk):
                nbs = [t[s:s + chunk] for t in self.tables]
                h_nbs = [prev[nb] for nb in nbs]
                s_nbs = [scores[level][nb] for nb in nbs]
                out.append(layer(prev[s:s + chunk], h_nbs, self._p(level), s_nbs)[0])
            views.append(torch.cat(out))
        self.ref_views = views
        self.ref_scores = scores

    @torch.no_grad()
    def _embed_new(self, Z: np.ndarray) -> torch.Tensor:
        nbs = [torch.as_tensor(ix.query(Z[:, v]), dtype=torch.long)
               for ix, v in zip(self.indices, self.views_)]
        h = self._to_tensor(Z)
        for level, layer in enumerate(self.net.layers):
            h_nbs = [self.ref_views[level][nb] for nb in nbs]
            s_nbs = [self.ref_scores[level][nb] for nb in nbs]
            h = layer(h, h_nbs, self._p(level), s_nbs)[0]
        return h

    @torch.no_grad()
    def _predict_fraud_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        Z = self.scaler.transform(X)
        out = []
        for s in range(0, Z.shape[0], self.config.inference_chunk):
            h = self._embed_new(Z[s:s + self.config.inference_chunk])
            out.append(torch.softmax(self.net.classifier(h), dim=1)[:, 1])
        return torch.cat(out).double().numpy()

    def embed(self, X: np.ndarray) -> np.ndarray:
        """Final CARE-GNN node embeddings ``h^(L)`` of (new) transactions."""
        self._check_fitted()
        return self._embed_new(self.scaler.transform(check_X(X, self.n_features_in_))).numpy()
