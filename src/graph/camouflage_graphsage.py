"""
src/graph/camouflage_graphsage.py
=================================

Camouflage-Resistant Temporal-Memory GraphSAGE (MG-ECFM Stage 2A).

Pipeline for a batch of transactions ``e_t = (u_t, v_t, x_t, A_t, tau_t)``:

1. **Temporal memory.** Each endpoint gets an *event node* whose state is the
   decayed pre-batch memory plus the event message,
   ``s_u(t) = s_u(t^-) e^{-delta dtau} + W_mem [x_t || A_t]``
   (:class:`~src.graph.temporal_memory.TemporalNodeMemory`).
2. **Inductive neighbourhood.** Historical neighbours come from a bounded
   recency index (O(1) insert, O(fanout) read): ``fanouts[0]`` most recent
   counterparties of ``u`` / ``v`` and ``fanouts[l]`` of every hop-``l``
   node. The current counterparty edge ``u <-> v`` is included - it is known
   at authorisation time. Historical nodes carry their decayed memory
   ``s_w(t)``; no label or future information is used as an input.
3. **Camouflage filter.** Every edge is scored by
   ``S(u, w) = sigmoid(W_sim |s_u - s_w|)`` and pruned below ``p``
   (:class:`~src.graph.similarity_filter.CamouflageSimilarityFilter`).
4. **GraphSAGE.** On the filtered neighbourhood ``N~(u)``:
   ``h_u^{(l)} = ReLU(W^{(l)} CONCAT(h_u^{(l-1)}, AGG{h_w^{(l-1)} : w in N~(u)}))``
   with ``h^{(0)} = s(t)``. Aggregation runs through a PyTorch Geometric
   ``MessagePassing`` operator when PyG is installed and an exactly
   equivalent ``index_add_`` / ``scatter_reduce`` kernel otherwise.
5. **Commit.** After scoring, memory and the recency index are updated so the
   next transaction sees this one. In stream mode commits happen at step
   boundaries (events of one step never see each other - the Phase-1
   causality contract); :meth:`CamouflageResistantGraphSAGE.embed_event` is
   the single-transaction authorisation fast path (sub-25 ms).

Training (:meth:`fit_stream`) is chronological, one pass per epoch with a
fresh memory: node-classification BCE on ``[h_u || h_v]`` plus the
label-aware similarity loss. Labels are used only as training targets,
never as model inputs.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.graph.similarity_filter import (
    CamouflageSimilarityFilter,
    FilterResult,
    SimilarityFilterConfig,
)
from src.graph.temporal_memory import TemporalMemoryConfig, TemporalNodeMemory

SEED = 42
MISSING_ENTITY = ""

try:  # PyTorch Geometric is optional; the fallback is numerically identical.
    from torch_geometric.nn import MessagePassing as _PyGMessagePassing

    HAS_PYG = True
except Exception:  # pragma: no cover - depends on the environment
    _PyGMessagePassing = None
    HAS_PYG = False


# --------------------------------------------------------------------------- #
# Neighbourhood aggregation                                                   #
# --------------------------------------------------------------------------- #
if HAS_PYG:

    class _PyGNeighborAggregator(_PyGMessagePassing):
        """PyG operator returning ``AGG{x_w : (w -> u) in edge_index}`` per node."""

        def __init__(self, aggr: str) -> None:
            super().__init__(aggr=aggr, flow="source_to_target")

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            n = x.size(0)
            return self.propagate(edge_index, x=x, size=(n, n))

        def message(self, x_j: torch.Tensor) -> torch.Tensor:  # noqa: D401
            return x_j


def scatter_aggregate(x: torch.Tensor, edge_index: torch.Tensor, aggr: str) -> torch.Tensor:
    """Pure-PyTorch neighbour aggregation (empty neighbourhoods -> 0)."""
    n, d = x.shape
    out = torch.zeros(n, d, dtype=x.dtype)
    if edge_index.numel() == 0:
        return out
    src, dst = edge_index[0], edge_index[1]
    msgs = x[src]
    if aggr in ("mean", "sum"):
        out = out.index_add(0, dst, msgs)
        if aggr == "mean":
            cnt = torch.bincount(dst, minlength=n).clamp_min(1).to(x.dtype)
            out = out / cnt.unsqueeze(1)
        return out
    if aggr == "max":
        idx = dst.unsqueeze(1).expand(-1, d)
        out = out.scatter_reduce(0, idx, msgs, reduce="amax", include_self=False)
        return out
    raise ValueError(f"Unsupported aggregator '{aggr}'")


class FilteredSAGEConv(nn.Module):
    """``h_u = ReLU(W CONCAT(x_u, AGG{x_w : w in N~(u)}))`` on a pruned graph."""

    def __init__(self, in_dim: int, out_dim: int, aggr: str = "mean",
                 use_pyg: Optional[bool] = None, activation: bool = True) -> None:
        super().__init__()
        if aggr not in ("mean", "sum", "max"):
            raise ValueError("aggr must be 'mean', 'sum' or 'max'")
        self.in_dim, self.out_dim, self.aggr = int(in_dim), int(out_dim), aggr
        self.lin = nn.Linear(2 * self.in_dim, self.out_dim)
        self.activation = bool(activation)
        self.use_pyg = HAS_PYG if use_pyg is None else bool(use_pyg and HAS_PYG)
        self._pyg = _PyGNeighborAggregator(aggr) if self.use_pyg else None

    def aggregate(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.zeros_like(x)
        if self._pyg is not None:
            return self._pyg(x, edge_index)
        return scatter_aggregate(x, edge_index, self.aggr)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        agg = self.aggregate(x, edge_index)
        out = self.lin(torch.cat([x, agg], dim=1))
        return F.relu(out) if self.activation else out


# --------------------------------------------------------------------------- #
# Recency neighbour index                                                     #
# --------------------------------------------------------------------------- #
class TemporalNeighborIndex:
    """Bounded most-recent-counterparty lists per node (deterministic sampling)."""

    def __init__(self, max_neighbors: int) -> None:
        if max_neighbors <= 0:
            raise ValueError("max_neighbors must be positive")
        self.max_neighbors = int(max_neighbors)
        self._adj: Dict[int, Deque[int]] = {}

    def __len__(self) -> int:
        return len(self._adj)

    def reset(self) -> None:
        self._adj = {}

    def insert(self, a: int, b: int) -> None:
        """Insert the undirected interaction ``a <-> b`` (O(1))."""
        dq = self._adj.get(a)
        if dq is None:
            dq = deque(maxlen=self.max_neighbors)
            self._adj[a] = dq
        dq.append(b)
        dq = self._adj.get(b)
        if dq is None:
            dq = deque(maxlen=self.max_neighbors)
            self._adj[b] = dq
        dq.append(a)

    def insert_many(self, src: Sequence[int], dst: Sequence[int]) -> None:
        for a, b in zip(src, dst):
            if a >= 0 and b >= 0:
                self.insert(int(a), int(b))

    def neighbors(self, node: int, k: int) -> List[int]:
        """Up to ``k`` most recent distinct counterparties, newest first."""
        dq = self._adj.get(node)
        if not dq:
            return []
        out: List[int] = []
        seen = set()
        for w in reversed(dq):
            if w in seen:
                continue
            seen.add(w)
            out.append(w)
            if len(out) >= k:
                break
        return out


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GraphSAGEConfig:
    """Hyper-parameters of :class:`CamouflageResistantGraphSAGE`."""

    feature_dim: int
    memory_dim: int = 64
    hidden_dim: int = 128
    out_dim: int = 64
    num_layers: int = 2
    fanouts: Tuple[int, ...] = (15, 10)
    aggregator: str = "mean"
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 20
    max_batch_events: int = 1024
    sim_loss_weight: float = 1.0
    max_pos_weight: float = 100.0
    latency_budget_ms: float = 25.0
    use_pyg: Optional[bool] = None
    memory: TemporalMemoryConfig = field(default=None)  # type: ignore[assignment]
    similarity: SimilarityFilterConfig = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        fan = tuple(int(f) for f in self.fanouts)
        if not fan or any(f <= 0 for f in fan):
            raise ValueError("fanouts must be positive integers")
        object.__setattr__(self, "fanouts", fan)
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.memory is None:
            object.__setattr__(self, "memory", TemporalMemoryConfig(
                feature_dim=self.feature_dim, memory_dim=self.memory_dim))
        if self.similarity is None:
            object.__setattr__(self, "similarity", SimilarityFilterConfig(
                memory_dim=self.memory_dim))
        if self.memory.memory_dim != self.memory_dim or self.similarity.memory_dim != self.memory_dim:
            raise ValueError("memory_dim must agree across memory / similarity configs")
        if self.memory.feature_dim != self.feature_dim:
            raise ValueError("feature_dim must agree with the memory config")

    def fanout(self, hop: int) -> int:
        return self.fanouts[min(hop, len(self.fanouts) - 1)]

    @classmethod
    def from_hyperparams(cls, hyperparams: Mapping, feature_dim: int,
                         **overrides) -> "GraphSAGEConfig":
        gs = hyperparams.get("graphsage", {}) if hyperparams else {}
        mem = TemporalMemoryConfig.from_hyperparams(hyperparams, feature_dim)
        sim = SimilarityFilterConfig.from_hyperparams(hyperparams, mem.memory_dim)
        kwargs = dict(
            feature_dim=int(feature_dim),
            memory_dim=mem.memory_dim,
            hidden_dim=int(gs.get("hidden_dim", 128)),
            out_dim=int(gs.get("out_dim", 64)),
            num_layers=int(gs.get("num_layers", 2)),
            fanouts=tuple(gs.get("fanouts", (15, 10))),
            aggregator=str(gs.get("aggregator", "mean")),
            dropout=float(gs.get("dropout", 0.2)),
            learning_rate=float(gs.get("learning_rate", 1e-3)),
            weight_decay=float(gs.get("weight_decay", 5e-4)),
            epochs=int(gs.get("epochs", 20)),
            max_batch_events=int(gs.get("batch_size", 1024)),
            latency_budget_ms=float(gs.get("latency_budget_ms", 25.0)),
            memory=mem,
            similarity=sim,
        )
        kwargs.update(overrides)
        return cls(**kwargs)


@dataclass
class SubgraphBatch:
    """Local computation graph for a batch of ``n`` events."""

    n_events: int
    states: torch.Tensor            # (2n + H, d_s) layer-0 node states
    edge_index: torch.Tensor        # (2, E) neighbour -> centre
    hist_global: List[int]          # global ids of the H historical nodes
    src_global: torch.Tensor        # (n,) global ids of event sources
    dst_global: torch.Tensor        # (n,) global ids of event destinations (-1 missing)
    message_input: torch.Tensor     # (n, message_dim)
    times: torch.Tensor             # (n,) event times


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
class CamouflageResistantGraphSAGE(nn.Module):
    """Temporal-memory GraphSAGE over camouflage-filtered neighbourhoods."""

    def __init__(self, config: GraphSAGEConfig, seed: int = SEED) -> None:
        super().__init__()
        self.config = config
        self.seed = int(seed)
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            self.memory = TemporalNodeMemory(config.memory, seed=self.seed)
            self.similarity = CamouflageSimilarityFilter(config.similarity, seed=self.seed + 1)
            dims = [config.memory_dim] + [config.hidden_dim] * (config.num_layers - 1) + [config.out_dim]
            self.convs = nn.ModuleList([
                FilteredSAGEConv(dims[i], dims[i + 1], aggr=config.aggregator,
                                 use_pyg=config.use_pyg)
                for i in range(config.num_layers)
            ])
            self.head = nn.Linear(2 * config.out_dim, 1)
        self.dropout = nn.Dropout(config.dropout)
        self.neighbors = TemporalNeighborIndex(max(config.fanouts))
        self._node_label: Dict[int, int] = {}
        self.training_history: List[Dict[str, float]] = []
        self.last_filter_stats: Dict[str, float] = {}

    @property
    def out_dim(self) -> int:
        return int(self.config.out_dim)

    # --------------------------------------------------------------- state
    def reset_state(self) -> None:
        """Forget node memories, neighbourhoods and observed node labels."""
        self.memory.reset_state()
        self.neighbors.reset()
        self._node_label = {}

    def fit_normalizer(self, features: np.ndarray, amounts: np.ndarray) -> None:
        self.memory.fit_normalizer(features, amounts)

    # ------------------------------------------------------- key handling
    def _register_events(self, src_keys: Sequence[Hashable],
                         dst_keys: Sequence[Hashable]) -> Tuple[torch.Tensor, torch.Tensor]:
        src = self.memory.register(src_keys)
        dst_list: List[int] = []
        for key in dst_keys:
            if _is_missing(key):
                dst_list.append(-1)
            else:
                dst_list.append(self.memory.register_one(key))
        return src, torch.as_tensor(dst_list, dtype=torch.long)

    # -------------------------------------------------- subgraph builder
    def build_subgraph(self, src_global: torch.Tensor, dst_global: torch.Tensor,
                       message_input: torch.Tensor, times: torch.Tensor) -> SubgraphBatch:
        """Assemble event nodes, multi-hop historical neighbours and edges."""
        cfg = self.config
        n = int(src_global.shape[0])
        hist_local: Dict[int, int] = {}
        hist_global: List[int] = []
        edge_src: List[int] = []
        edge_dst: List[int] = []

        def local_hist(g: int) -> int:
            loc = hist_local.get(g)
            if loc is None:
                loc = 2 * n + len(hist_global)
                hist_local[g] = loc
                hist_global.append(g)
            return loc

        src_list = src_global.tolist()
        dst_list = dst_global.tolist()
        frontier: List[int] = []
        k0 = cfg.fanout(0)
        for i in range(n):
            u, v = src_list[i], dst_list[i]
            for w in self.neighbors.neighbors(u, k0):
                edge_src.append(local_hist(w))
                edge_dst.append(i)
            if v >= 0:
                edge_src.append(n + i)
                edge_dst.append(i)
                edge_src.append(i)
                edge_dst.append(n + i)
                for w in self.neighbors.neighbors(v, k0):
                    edge_src.append(local_hist(w))
                    edge_dst.append(n + i)
        frontier = list(hist_global)
        expanded = set()
        for hop in range(1, cfg.num_layers):
            k = cfg.fanout(hop)
            next_frontier: List[int] = []
            for g in frontier:
                if g in expanded:
                    continue
                expanded.add(g)
                centre = hist_local[g]
                for w in self.neighbors.neighbors(g, k):
                    is_new = w not in hist_local
                    edge_src.append(local_hist(w))
                    edge_dst.append(centre)
                    if is_new:
                        next_frontier.append(w)
            frontier = next_frontier

        t_ref = float(times.max()) if n else 0.0
        src_states = self.memory.event_states(src_global, times, message_input)
        dst_states = torch.zeros(n, cfg.memory_dim, dtype=src_states.dtype)
        valid = dst_global >= 0
        if bool(valid.any()):
            vidx = torch.nonzero(valid).squeeze(1)
            dst_part = self.memory.event_states(dst_global[vidx], times[vidx], message_input[vidx])
            dst_states = dst_states.index_put((vidx,), dst_part)
        if hist_global:
            hist_states = self.memory.read(torch.as_tensor(hist_global, dtype=torch.long), t_ref)
        else:
            hist_states = torch.zeros(0, cfg.memory_dim, dtype=src_states.dtype)
        states = torch.cat([src_states, dst_states, hist_states], dim=0)
        edge_index = torch.as_tensor([edge_src, edge_dst], dtype=torch.long).reshape(2, -1)
        return SubgraphBatch(n, states, edge_index, hist_global, src_global, dst_global,
                             message_input, times)

    # ------------------------------------------------------------ forward
    def propagate(self, states: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, FilterResult]:
        """Camouflage filtering followed by the GraphSAGE layer stack."""
        filt = self.similarity.filter_edges(states, edge_index, n_nodes=states.shape[0])
        h = states
        for layer, conv in enumerate(self.convs):
            h = conv(h, filt.edge_index)
            if layer < len(self.convs) - 1:
                h = self.dropout(h)
        return h, filt

    def forward_subgraph(self, batch: SubgraphBatch) -> Dict[str, torch.Tensor]:
        h, filt = self.propagate(batch.states, batch.edge_index)
        n = batch.n_events
        h_src, h_dst = h[:n], h[n:2 * n]
        logits = self.head(torch.cat([h_src, h_dst], dim=1)).squeeze(1)
        total = int(filt.keep_mask.numel())
        self.last_filter_stats = {
            "n_edges": float(total),
            "n_kept": float(int(filt.keep_mask.sum())) if total else 0.0,
            "kept_fraction": float(filt.keep_mask.float().mean()) if total else 1.0,
        }
        return {"h_src": h_src, "h_dst": h_dst, "logits": logits, "filter": filt}

    def commit(self, batch: SubgraphBatch, labels: Optional[np.ndarray] = None,
               insert_edges: bool = True) -> None:
        """Write the batch into memory and the recency index (after scoring)."""
        with torch.no_grad():
            self.memory.commit(batch.src_global, batch.times, batch.message_input)
            valid = batch.dst_global >= 0
            if bool(valid.any()):
                vidx = torch.nonzero(valid).squeeze(1)
                self.memory.commit(batch.dst_global[vidx], batch.times[vidx],
                                   batch.message_input[vidx])
        if insert_edges:
            self.neighbors.insert_many(batch.src_global.tolist(), batch.dst_global.tolist())
        if labels is not None:
            for u, v, y in zip(batch.src_global.tolist(), batch.dst_global.tolist(),
                               np.asarray(labels).astype(int).tolist()):
                if y:
                    self._node_label[u] = 1
                    if v >= 0:
                        self._node_label[v] = 1

    def commit_step(self, pending: Sequence[Tuple[SubgraphBatch, Optional[np.ndarray]]]) -> None:
        """Commit every batch of one step, inserting edges in canonical key order.

        Memory commits are order-invariant (closed form). Recency-index
        insertions are sorted by ``(src_key, dst_key)`` so that the sampled
        neighbourhoods - and hence embeddings - are invariant to the arrival
        order of transactions inside a step.
        """
        pairs: List[Tuple[str, str, int, int]] = []
        for batch, labels in pending:
            self.commit(batch, labels, insert_edges=False)
            for u, v in zip(batch.src_global.tolist(), batch.dst_global.tolist()):
                if v >= 0:
                    pairs.append((str(self.memory.key_of(u)), str(self.memory.key_of(v)), u, v))
        pairs.sort()
        self.neighbors.insert_many([p[2] for p in pairs], [p[3] for p in pairs])

    # ----------------------------------------------------------- training
    def _similarity_loss(self, batch: SubgraphBatch, labels: torch.Tensor) -> torch.Tensor:
        if batch.edge_index.numel() == 0:
            return torch.zeros(())
        n = batch.n_events
        hist_labels = [self._node_label.get(g, 0) for g in batch.hist_global]
        node_y = torch.cat([labels, labels,
                            torch.as_tensor(hist_labels, dtype=labels.dtype)])
        states = batch.states.detach()
        ei = batch.edge_index
        return self.similarity.label_aware_loss(states[ei[1]], states[ei[0]],
                                                node_y[ei[1]], node_y[ei[0]])

    def fit_stream(self, features: np.ndarray, amounts: np.ndarray, steps: np.ndarray,
                   src_keys: Sequence[Hashable], dst_keys: Sequence[Hashable],
                   labels: np.ndarray, epochs: Optional[int] = None,
                   verbose: bool = False) -> List[Dict[str, float]]:
        """Chronological mini-batch training; memory is rebuilt every epoch.

        Within a step all mini-batches read the pre-step state; commits happen
        once the whole step has been processed.
        """
        cfg = self.config
        steps = np.asarray(steps)
        _check_sorted(steps)
        labels = np.asarray(labels).astype(np.float32)
        if not self.memory.feature_scaler.is_fitted:
            self.fit_normalizer(features, amounts)
        msg_all = self.memory.build_message_input(features, amounts)
        pos = float(labels.sum())
        neg = float(labels.shape[0] - pos)
        pos_weight = torch.tensor(min(cfg.max_pos_weight, neg / max(pos, 1.0)))
        opt = torch.optim.Adam(self.parameters(), lr=cfg.learning_rate,
                               weight_decay=cfg.weight_decay)
        torch.manual_seed(self.seed)
        n_epochs = int(epochs if epochs is not None else cfg.epochs)
        history: List[Dict[str, float]] = []
        src_keys = list(src_keys)
        dst_keys = list(dst_keys)
        bounds = _step_bounds(steps)
        for epoch in range(n_epochs):
            self.train()
            self.reset_state()
            tot_cls = tot_sim = 0.0
            n_batches = 0
            for start, end in bounds:
                pending: List[Tuple[SubgraphBatch, np.ndarray]] = []
                for b0 in range(start, end, cfg.max_batch_events):
                    b1 = min(end, b0 + cfg.max_batch_events)
                    src_g, dst_g = self._register_events(src_keys[b0:b1], dst_keys[b0:b1])
                    times = torch.full((b1 - b0,), float(steps[b0]), dtype=torch.float64)
                    batch = self.build_subgraph(src_g, dst_g, msg_all[b0:b1], times)
                    y = torch.from_numpy(labels[b0:b1])
                    out = self.forward_subgraph(batch)
                    cls_loss = F.binary_cross_entropy_with_logits(
                        out["logits"], y, pos_weight=pos_weight)
                    sim_loss = self._similarity_loss(batch, y)
                    loss = cls_loss + cfg.sim_loss_weight * sim_loss
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    tot_cls += float(cls_loss.detach())
                    tot_sim += float(sim_loss.detach())
                    n_batches += 1
                    pending.append((batch, labels[b0:b1]))
                self.commit_step(pending)
            record = {"epoch": float(epoch), "cls_loss": tot_cls / max(n_batches, 1),
                      "sim_loss": tot_sim / max(n_batches, 1)}
            history.append(record)
            if verbose:
                print(f"[graphsage] epoch {epoch}: cls={record['cls_loss']:.4f} "
                      f"sim={record['sim_loss']:.4f}")
        self.eval()
        self.training_history.extend(history)
        return history

    # ---------------------------------------------------------- inference
    @torch.no_grad()
    def transform_stream(self, features: np.ndarray, amounts: np.ndarray, steps: np.ndarray,
                         src_keys: Sequence[Hashable], dst_keys: Sequence[Hashable],
                         labels: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Embeddings ``h_u, h_v`` (and GNN fraud probabilities) for a stream.

        Continues from the current memory; commits at every step boundary.
        ``labels`` (optional) are only recorded as observed node labels for a
        later similarity-loss fine-tune - they never influence the output.
        """
        self.eval()
        cfg = self.config
        steps = np.asarray(steps)
        _check_sorted(steps)
        n = steps.shape[0]
        h_src = np.zeros((n, cfg.out_dim), dtype=np.float32)
        h_dst = np.zeros((n, cfg.out_dim), dtype=np.float32)
        prob = np.zeros(n, dtype=np.float64)
        if n == 0:
            return {"h_src": h_src, "h_dst": h_dst, "gnn_proba": prob}
        msg_all = self.memory.build_message_input(features, amounts)
        src_keys = list(src_keys)
        dst_keys = list(dst_keys)
        for start, end in _step_bounds(steps):
            pending: List[Tuple[SubgraphBatch, Optional[np.ndarray]]] = []
            for b0 in range(start, end, cfg.max_batch_events):
                b1 = min(end, b0 + cfg.max_batch_events)
                src_g, dst_g = self._register_events(src_keys[b0:b1], dst_keys[b0:b1])
                times = torch.full((b1 - b0,), float(steps[b0]), dtype=torch.float64)
                batch = self.build_subgraph(src_g, dst_g, msg_all[b0:b1], times)
                out = self.forward_subgraph(batch)
                h_src[b0:b1] = out["h_src"].numpy()
                h_dst[b0:b1] = out["h_dst"].numpy()
                prob[b0:b1] = torch.sigmoid(out["logits"]).double().numpy()
                pending.append((batch, None if labels is None else np.asarray(labels)[b0:b1]))
            self.commit_step(pending)
        return {"h_src": h_src, "h_dst": h_dst, "gnn_proba": prob}

    @torch.no_grad()
    def embed_event(self, src_key: Hashable, dst_key: Hashable, features: np.ndarray,
                    amount: float, step: float, commit: bool = True) -> Dict[str, object]:
        """Single-transaction authorisation fast path.

        Returns ``h_u``, ``h_v`` (NumPy, ``(out_dim,)``), the GNN probability
        and the wall-clock latency in milliseconds.
        """
        t0 = time.perf_counter()
        self.eval()
        u = self.memory.register_one(src_key)
        v = -1 if _is_missing(dst_key) else self.memory.register_one(dst_key)
        msg = self.memory.build_message_input(np.asarray(features, dtype=np.float64)[None, :],
                                              np.asarray([amount], dtype=np.float64))
        times = torch.tensor([float(step)], dtype=torch.float64)
        batch = self.build_subgraph(torch.tensor([u]), torch.tensor([v]), msg, times)
        out = self.forward_subgraph(batch)
        if commit:
            self.commit(batch)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "h_src": out["h_src"][0].numpy().copy(),
            "h_dst": out["h_dst"][0].numpy().copy(),
            "gnn_proba": float(torch.sigmoid(out["logits"][0])),
            "latency_ms": latency_ms,
            "n_edges": int(batch.edge_index.shape[1]),
            "n_kept_edges": int(out["filter"].keep_mask.sum()) if batch.edge_index.numel() else 0,
        }


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _is_missing(key: Hashable) -> bool:
    if key is None:
        return True
    if isinstance(key, float) and math.isnan(key):
        return True
    return isinstance(key, str) and key == MISSING_ENTITY


def _check_sorted(steps: np.ndarray) -> None:
    if steps.size > 1 and np.any(np.diff(steps) < 0):
        raise ValueError("Events must be sorted chronologically by step")


def _step_bounds(steps: np.ndarray) -> List[Tuple[int, int]]:
    if steps.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(steps)) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [steps.size]))
    return [(int(s), int(e)) for s, e in zip(starts, ends)]
