"""
src/graph/similarity_filter.py
==============================

Label-aware camouflage filter applied *before* GraphSAGE message passing.

For a centre node ``u`` and neighbour ``w`` with temporal states ``s_u(t)``
and ``s_w(t)`` the edge similarity is

    S(u, w) = sigmoid( W_sim |s_u(t) - s_w(t)| )                  (l1 / CARE-GNN)

(an optional hidden layer turns ``W_sim`` into a 1-hidden-layer MLP). The
scorer is trained *label-aware*: an edge is a positive (target 1) when both
endpoints share a class and a negative (target 0) when the edge crosses the
fraud/legit boundary - exactly the relation camouflage a money mule creates by
transacting with many benign accounts. Class-balanced weights stop the
dominant benign-benign edges from swamping the camouflage signal.

Two pruning modes construct the filtered neighbourhood ``N~(u)``:

* ``threshold`` - keep ``w`` iff ``S(u, w) >= p``;
* ``top_p``     - keep the ``ceil(p * |N(u)|)`` most similar neighbours (CARE-GNN).

``min_neighbors_kept`` (default 0 in threshold mode semantics, configurable)
retains the most similar neighbours of a node whose entire neighbourhood would
otherwise be pruned. All operations are vectorised: per-node ranking uses two
stable sorts, so the cost is O(E log E) with no Python loops.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, NamedTuple, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SEED = 42


@dataclass(frozen=True)
class SimilarityFilterConfig:
    """Hyper-parameters of :class:`CamouflageSimilarityFilter`."""

    memory_dim: int
    hidden_dim: int = 0
    threshold_p: float = 0.5
    mode: str = "threshold"
    min_neighbors_kept: int = 0
    learning_rate: float = 1e-3
    distance: str = "l1"

    def __post_init__(self) -> None:
        if self.memory_dim <= 0:
            raise ValueError("memory_dim must be positive")
        if self.hidden_dim < 0:
            raise ValueError("hidden_dim must be >= 0")
        if not 0.0 <= self.threshold_p <= 1.0:
            raise ValueError("threshold_p must lie in [0, 1]")
        if self.mode not in ("threshold", "top_p"):
            raise ValueError("mode must be 'threshold' or 'top_p'")
        if self.min_neighbors_kept < 0:
            raise ValueError("min_neighbors_kept must be >= 0")
        if self.distance not in ("l1", "l2"):
            raise ValueError("distance must be 'l1' or 'l2'")

    @classmethod
    def from_hyperparams(cls, hyperparams: Mapping, memory_dim: int) -> "SimilarityFilterConfig":
        sf = hyperparams.get("similarity_filter", {}) if hyperparams else {}
        return cls(
            memory_dim=int(memory_dim),
            hidden_dim=int(sf.get("hidden_dim", 0)),
            threshold_p=float(sf.get("similarity_threshold_p", 0.5)),
            mode=str(sf.get("mode", "threshold")),
            min_neighbors_kept=int(sf.get("min_neighbors_kept", 0)),
            learning_rate=float(sf.get("learning_rate", 1e-3)),
            distance=str(sf.get("distance", "l1")),
        )


class FilterResult(NamedTuple):
    """Output of :meth:`CamouflageSimilarityFilter.filter_edges`."""

    edge_index: torch.Tensor      # (2, E_kept) filtered edges (row 0 = neighbour, row 1 = centre)
    similarity: torch.Tensor      # (E,) similarity of every input edge
    keep_mask: torch.Tensor       # (E,) bool mask of kept edges
    kept_similarity: torch.Tensor  # (E_kept,) similarity of kept edges


class CamouflageSimilarityFilter(nn.Module):
    """``S(u, w) = sigmoid(W_sim |s_u - s_w|)`` with label-aware training."""

    def __init__(self, config: SimilarityFilterConfig, seed: int = SEED) -> None:
        super().__init__()
        self.config = config
        gen = torch.Generator().manual_seed(int(seed))
        d = config.memory_dim
        if config.hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(d, config.hidden_dim), nn.ReLU(), nn.Linear(config.hidden_dim, 1)
            )
        else:
            self.net = nn.Sequential(nn.Linear(d, 1))
        with torch.no_grad():
            for module in self.net:
                if isinstance(module, nn.Linear):
                    bound = math.sqrt(6.0 / (module.in_features + module.out_features))
                    module.weight.copy_(
                        torch.rand(module.weight.shape, generator=gen) * 2 * bound - bound
                    )
                    module.bias.zero_()

    # ------------------------------------------------------------- scoring
    def distance(self, s_u: torch.Tensor, s_w: torch.Tensor) -> torch.Tensor:
        diff = s_u - s_w
        return diff.abs() if self.config.distance == "l1" else diff.pow(2)

    def logits(self, s_u: torch.Tensor, s_w: torch.Tensor) -> torch.Tensor:
        return self.net(self.distance(s_u, s_w)).squeeze(-1)

    def forward(self, s_u: torch.Tensor, s_w: torch.Tensor) -> torch.Tensor:
        """Similarity ``S(u, w)`` in (0, 1) for aligned rows of ``s_u`` and ``s_w``."""
        return torch.sigmoid(self.logits(s_u, s_w))

    def edge_similarity(self, states: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Similarity of each edge ``(w -> u)`` of ``edge_index`` (row0 = w, row1 = u)."""
        if edge_index.numel() == 0:
            return torch.zeros(0, dtype=states.dtype, device=states.device)
        return self.forward(states[edge_index[1]], states[edge_index[0]])

    # ------------------------------------------------------------- pruning
    @staticmethod
    def rank_within_centre(centres: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """0-based rank of every edge among edges sharing its centre (desc. score)."""
        e = centres.shape[0]
        if e == 0:
            return torch.zeros(0, dtype=torch.long, device=centres.device)
        order = torch.sort(scores, descending=True, stable=True).indices
        order = order[torch.sort(centres[order], stable=True).indices]
        sorted_centres = centres[order]
        positions = torch.arange(e, dtype=torch.long, device=centres.device)
        is_start = torch.ones(e, dtype=torch.bool, device=centres.device)
        is_start[1:] = sorted_centres[1:] != sorted_centres[:-1]
        group_start = torch.cummax(torch.where(is_start, positions, torch.zeros_like(positions)), 0).values
        ranks_sorted = positions - group_start
        ranks = torch.empty(e, dtype=torch.long, device=centres.device)
        ranks[order] = ranks_sorted
        return ranks

    def keep_mask(self, centres: torch.Tensor, scores: torch.Tensor,
                  n_nodes: Optional[int] = None) -> torch.Tensor:
        """Boolean mask of edges retained under the configured pruning rule."""
        cfg = self.config
        e = centres.shape[0]
        if e == 0:
            return torch.zeros(0, dtype=torch.bool, device=centres.device)
        n = int(n_nodes) if n_nodes is not None else int(centres.max()) + 1
        ranks = self.rank_within_centre(centres, scores)
        degree = torch.bincount(centres, minlength=n)
        if cfg.mode == "threshold":
            keep = scores >= cfg.threshold_p
        else:
            k_keep = torch.ceil(degree.to(torch.float64) * cfg.threshold_p).to(torch.long)
            keep = ranks < k_keep[centres]
        if cfg.min_neighbors_kept > 0:
            keep = keep | (ranks < cfg.min_neighbors_kept)
        return keep

    def filter_edges(self, states: torch.Tensor, edge_index: torch.Tensor,
                     n_nodes: Optional[int] = None) -> FilterResult:
        """Prune camouflaged edges; returns the filtered ``edge_index`` and scores."""
        edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=states.device)
        if edge_index.numel() == 0:
            empty = torch.zeros(0, dtype=states.dtype, device=states.device)
            return FilterResult(edge_index.reshape(2, 0), empty,
                                torch.zeros(0, dtype=torch.bool, device=states.device), empty)
        sim = self.edge_similarity(states, edge_index)
        keep = self.keep_mask(edge_index[1], sim.detach(),
                              n_nodes if n_nodes is not None else states.shape[0])
        return FilterResult(edge_index[:, keep], sim, keep, sim[keep])

    # ------------------------------------------------------------- training
    @staticmethod
    def label_aware_targets(y_centre: torch.Tensor, y_neigh: torch.Tensor) -> torch.Tensor:
        """1 for same-class edges, 0 for cross-class (camouflage) edges."""
        return (y_centre.to(torch.long) == y_neigh.to(torch.long)).to(torch.float32)

    def label_aware_loss(self, s_u: torch.Tensor, s_w: torch.Tensor,
                         y_u: torch.Tensor, y_w: torch.Tensor,
                         balance: bool = True) -> torch.Tensor:
        """Class-balanced BCE between ``S(u, w)`` and the same-class indicator."""
        if s_u.shape[0] == 0:
            return torch.zeros((), dtype=torch.float32, device=s_u.device, requires_grad=False)
        target = self.label_aware_targets(y_u, y_w)
        logit = self.logits(s_u, s_w)
        if balance:
            n_pos = target.sum()
            n_neg = target.numel() - n_pos
            w_pos = 0.5 / n_pos.clamp_min(1.0)
            w_neg = 0.5 / n_neg.clamp_min(1.0)
            if n_pos == 0 or n_neg == 0:
                weight = torch.full_like(target, 1.0 / target.numel())
            else:
                weight = torch.where(target > 0.5, w_pos, w_neg)
            return F.binary_cross_entropy_with_logits(logit, target, weight=weight, reduction="sum")
        return F.binary_cross_entropy_with_logits(logit, target)

    def fit(self, states: torch.Tensor, edge_index: torch.Tensor, node_labels: torch.Tensor,
            epochs: int = 200, learning_rate: Optional[float] = None,
            seed: int = SEED) -> Dict[str, float]:
        """Fit ``W_sim`` on a labelled static graph snapshot (full-batch Adam)."""
        torch.manual_seed(seed)
        states = states.detach()
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        y = torch.as_tensor(node_labels)
        opt = torch.optim.Adam(self.parameters(), lr=learning_rate or self.config.learning_rate)
        s_u, s_w = states[edge_index[1]], states[edge_index[0]]
        y_u, y_w = y[edge_index[1]], y[edge_index[0]]
        first = last = float("nan")
        for epoch in range(int(epochs)):
            opt.zero_grad()
            loss = self.label_aware_loss(s_u, s_w, y_u, y_w)
            loss.backward()
            opt.step()
            if epoch == 0:
                first = float(loss.detach())
            last = float(loss.detach())
        return {"initial_loss": first, "final_loss": last}

    @torch.no_grad()
    def camouflage_report(self, states: torch.Tensor, edge_index: torch.Tensor,
                          node_labels: torch.Tensor) -> Dict[str, float]:
        """Pruning statistics split by homophilic vs. camouflaged (cross-class) edges."""
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        y = torch.as_tensor(node_labels).to(torch.long)
        res = self.filter_edges(states, edge_index)
        cross = y[edge_index[0]] != y[edge_index[1]]
        n_cross = int(cross.sum())
        n_homo = int((~cross).sum())
        return {
            "n_edges": int(edge_index.shape[1]),
            "n_camouflage_edges": n_cross,
            "camouflage_pruned_rate": float((~res.keep_mask & cross).sum()) / max(n_cross, 1),
            "homophilic_kept_rate": float((res.keep_mask & ~cross).sum()) / max(n_homo, 1),
            "kept_fraction": float(res.keep_mask.float().mean()) if res.keep_mask.numel() else 0.0,
        }


def numpy_states(states: np.ndarray) -> torch.Tensor:
    """Convenience: float32 tensor view of a NumPy state matrix."""
    return torch.from_numpy(np.ascontiguousarray(states, dtype=np.float32))
