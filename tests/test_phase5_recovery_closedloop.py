"""
tests/test_phase5_recovery_closedloop.py
========================================

Phase 5 verification: Stage 5 (TD-TPF recovery) and Stage 6 (closed-loop
delayed feedback) of MG-ECFM.

1. TD-TPF computes the exact proportional taint across a 3-hop mule laundering
   chain (hand-computed), stops at CASH_OUT / external wires, conserves mass
   and builds identical graphs from PaySim, AMLSim and Phase 1 canonical frames.
2. Taint decays as ``exp(-kappa (tau' - tau))`` per lot; ``kappa = 0`` recovers
   pure proportional flow; decay is monotone in ``kappa`` and in elapsed time.
3. The recovery simulator reports exact Recovered Amount ($), Recovery Rate (%),
   Prevented-Loss Ratio (%) and net savings inside the 24 h window, follows funds
   that move during the reaction latency, and recovers nothing after the window.
4. The verification queue keeps labels hidden until ``tau + Delta_tau``
   (``Delta_tau in [1, 14]`` days) and releases them chronologically.
5. The closed-loop adapter updates the CS-ACRC thresholds ``lambda_{t+1}`` and the
   martingale wealth ``M_t`` only when labels are released, rolls back degraded
   rules and schedules retraining.

Run from the repository root:

    python tests/test_phase5_recovery_closedloop.py          # unittest runner
    python -m pytest tests/test_phase5_recovery_closedloop.py -q
"""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.closed_loop.feedback_adapter import (  # noqa: E402
    REASON_DRIFT,
    REASON_ROLLBACK,
    REASON_VOLUME,
    AdapterConfig,
    ClosedLoopAdapter,
)
from src.closed_loop.verification_queue import (  # noqa: E402
    SOURCE_CHARGEBACK,
    SOURCE_DISPUTE,
    SOURCE_MATURITY,
    TELEMETRY_BLOCK,
    TELEMETRY_STEP_UP_FAIL,
    TELEMETRY_STEP_UP_PASS,
    LatencyModel,
    VerificationQueue,
)
from src.conformal.adaptive_crc import OnlineAdaptiveCRC, project_ordered_thresholds  # noqa: E402
from src.conformal.loss_functions import Action, AsymmetricLoss, CostParameters  # noqa: E402
from src.data.perturbation_generator import CandidateRule, Condition, SourceEvidence  # noqa: E402
from src.epistemic_gate.martingale_eprocess import GateConfig, MartingaleEProcessGate  # noqa: E402
from src.recovery.action_simulator import (  # noqa: E402
    ACTION_BLOCK,
    ACTION_CHARGEBACK,
    ACTION_FREEZE,
    RecoveryPolicy,
    RecoverySimulator,
)
from src.recovery.tainted_flow import (  # noqa: E402
    EDGE_CASH_OUT,
    EDGE_EXTERNAL_WIRE,
    EDGE_INTERNAL,
    EDGE_MERCHANT,
    EdgeSchema,
    FlowEdge,
    TaintedFlowTracker,
)

SEED = 42
CONFIG_DIR = ROOT / "configs"
HYPERPARAMS = json.loads((CONFIG_DIR / "hyperparams.json").read_text(encoding="utf-8"))
PROJECT_CONFIG = json.loads((CONFIG_DIR / "project_config.json").read_text(encoding="utf-8"))
KAPPA = float(HYPERPARAMS["tdtpf"]["kappa_decay_per_hour"])
W_REC = float(HYPERPARAMS["tdtpf"]["recovery_window_hours"])
GATE_CFG = GateConfig.from_config(HYPERPARAMS)
TOL = 1e-9


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
A0 = 10_000.0
OPENING = {"M2": 2_000.0, "M3": 1_000.0}


def chain_edges(cash_out_tau: float = 5.0):
    """Victim U -> M1 -> M2 -> M3 -> CASH_OUT (3 mule hops after the beneficiary M1)."""
    return [
        FlowEdge(0, "U", "M1", A0, 0.0, "TRANSFER", 1),
        FlowEdge(1, "M1", "M2", 6_000.0, 1.0, "TRANSFER", 0),
        FlowEdge(2, "M2", "M3", 4_000.0, 3.0, "TRANSFER", 0),
        FlowEdge(3, "M3", "ATM_AGENT", 1_000.0, cash_out_tau, "CASH_OUT", 0),
    ]


def chain_hand(kappa: float):
    """Hand-computed TD-TPF quantities for :func:`chain_edges` (cash-out at tau = 5)."""
    s1 = 6_000.0 / ((A0 - 6_000.0) + 6_000.0)              # M1 post-balance 4000
    s2 = 4_000.0 / ((2_000.0 + 6_000.0 - 4_000.0) + 4_000.0)  # M2 pre-balance 8000
    s3 = 1_000.0 / ((1_000.0 + 4_000.0 - 1_000.0) + 1_000.0)  # M3 pre-balance 5000
    t_m2 = A0 * s1 * math.exp(-kappa * (1.0 - 0.0))
    t_m3 = t_m2 * s2 * math.exp(-kappa * (3.0 - 1.0))
    exited = t_m3 * s3 * math.exp(-kappa * (5.0 - 3.0))
    held = {"M1": A0 * (1 - s1), "M2": t_m2 * (1 - s2), "M3": t_m3 * (1 - s3)}
    return {"shares": (s1, s2, s3), "t_m2": t_m2, "t_m3": t_m3, "exited": exited, "held": held,
            "balances": {"M1": 4_000.0, "M2": 4_000.0, "M3": 4_000.0}}


def tracker(edges, **kw):
    kw.setdefault("kappa", KAPPA)
    kw.setdefault("opening_balances", OPENING)
    kw.setdefault("min_taint", 0.0)
    return TaintedFlowTracker(edges, kw.pop("schema", "generic"), **kw)


def make_rule(rule_id: str, precision: float) -> CandidateRule:
    return CandidateRule(rule_id, "genuine",
                         (Condition("log1p_amount", ">", 9.0), Condition("tx_type", "==", "TRANSFER")),
                         (SourceEvidence("analyst_alert_001", "analyst_alert", 0.9, 90, 10, 630, 500),),
                         emitted_at_step=630, true_precision=float(precision),
                         historical_precision=float(precision), is_harmful=precision <= GATE_CFG.p0)


def make_crc(**kw) -> OnlineAdaptiveCRC:
    kw.setdefault("warmup", 0)
    kw.setdefault("alpha", 0.02)
    kw.setdefault("gamma", 0.005)
    return OnlineAdaptiveCRC(loss=AsymmetricLoss(CostParameters.from_config(PROJECT_CONFIG)), **kw)


def random_network(rng: np.random.Generator, n_nodes: int = 12, n_edges: int = 60):
    nodes = [f"N{i}" for i in range(n_nodes)]
    edges = [FlowEdge(0, "VICTIM", "N0", 5_000.0, 0.0, "TRANSFER", 1)]
    taus = np.sort(rng.uniform(0.0, 20.0, n_edges))
    for k, t in enumerate(taus, start=1):
        a, b = rng.choice(n_nodes, size=2, replace=False)
        kind = rng.random()
        ttype = "CASH_OUT" if kind < 0.08 else ("WIRE" if kind < 0.12 else "TRANSFER")
        edges.append(FlowEdge(k, nodes[a], nodes[b], float(rng.uniform(50, 3_000)), float(t), ttype, 0))
    opening = {v: float(rng.uniform(0, 4_000)) for v in nodes}
    return edges, opening


# --------------------------------------------------------------------------- #
# 1. TD-TPF proportional taint                                                #
# --------------------------------------------------------------------------- #
class TestTaintedProportionalFlow(unittest.TestCase):

    def test_three_hop_mule_chain_hand_computed(self) -> None:
        hand = chain_hand(KAPPA)
        tr = tracker(chain_edges()).trace(0, tau_detect=10.0)
        taint = tr.node_taint()
        for v in ("M1", "M2", "M3"):
            self.assertAlmostEqual(taint[v], hand["held"][v], places=9)
            self.assertAlmostEqual(tr.balances[v], hand["balances"][v], places=9)
        # Edge-level factors: share = A'/(Balance(v_a)+A'), decay = exp(-kappa dt).
        ef = tr.edges_frame().set_index("edge_id")
        for eid, share in zip((1, 2, 3), hand["shares"]):
            self.assertAlmostEqual(ef.loc[eid, "share"], share, places=12)
        self.assertAlmostEqual(ef.loc[1, "delivered"], hand["t_m2"], places=9)
        self.assertAlmostEqual(ef.loc[2, "delivered"], hand["t_m3"], places=9)
        self.assertAlmostEqual(ef.loc[2, "decay"], math.exp(-KAPPA * 2.0), places=12)
        self.assertEqual(ef["hop"].tolist(), [1, 2, 3])
        # Cash-out terminates tracking and is booked as exited.
        self.assertEqual(ef.loc[3, "kind"], EDGE_CASH_OUT)
        self.assertAlmostEqual(tr.exited()[EDGE_CASH_OUT], hand["exited"], places=9)
        self.assertNotIn("ATM_AGENT", taint)
        # Freezable value min(T, Balance) and the derived metrics.
        liquid = tr.liquid_recoverable()
        for v in ("M1", "M2", "M3"):
            self.assertAlmostEqual(liquid[v], min(hand["held"][v], hand["balances"][v]), places=9)
        m = tr.metrics()
        recovered = sum(min(hand["held"][v], hand["balances"][v]) for v in ("M1", "M2", "M3"))
        self.assertAlmostEqual(m["recovered_amount"], recovered, places=9)
        self.assertAlmostEqual(m["recovery_rate_pct"], 100.0 * recovered / A0, places=9)
        self.assertEqual(tr.min_hop(), {"M1": 0, "M2": 1, "M3": 2})
        self.assertLess(abs(tr.conservation_residual()), TOL)
        self.assertEqual(tr.taint_concentration()["M1"], 1.0)

    def test_commingled_clean_inflow_dilutes_share(self) -> None:
        edges = chain_edges() + [FlowEdge(9, "CLEAN", "M2", 4_000.0, 2.0, "TRANSFER", 0)]
        tr = tracker(edges).trace(0, tau_detect=4.0)
        t_m2 = A0 * 0.6 * math.exp(-KAPPA * 1.0)
        share = 4_000.0 / (2_000.0 + 6_000.0 + 4_000.0)  # pre-balance 12 000 after clean deposit
        self.assertAlmostEqual(tr.edges_frame().set_index("edge_id").loc[2, "share"], share, places=12)
        self.assertAlmostEqual(tr.node_taint()["M3"], t_m2 * share * math.exp(-KAPPA * 2.0), places=9)
        self.assertNotIn("CLEAN", tr.node_taint())
        self.assertLess(abs(tr.conservation_residual()), TOL)

    def test_lots_decay_from_their_own_arrival_time(self) -> None:
        edges = [
            FlowEdge(0, "U", "A", 1_000.0, 0.0), FlowEdge(1, "A", "B", 500.0, 2.0),
            FlowEdge(2, "A", "B", 500.0, 5.0), FlowEdge(3, "B", "C", 1_000.0, 6.0),
        ]
        tr = tracker(edges, opening_balances={}).trace(0, tau_detect=8.0)
        lot1 = 500.0 * math.exp(-KAPPA * 2.0)          # share 0.5
        lot2 = 500.0 * math.exp(-KAPPA * 5.0)          # share 1.0 of remaining 500
        expected_c = lot1 * math.exp(-KAPPA * 4.0) + lot2 * math.exp(-KAPPA * 1.0)
        self.assertAlmostEqual(tr.node_taint()["C"], expected_c, places=9)
        self.assertNotIn("A", tr.node_taint())
        self.assertLess(abs(tr.conservation_residual()), TOL)

    def test_tracking_stops_at_cash_out_and_external_wire(self) -> None:
        edges = chain_edges() + [FlowEdge(4, "ATM_AGENT", "Z", 900.0, 6.0)]
        tr = tracker(edges).trace(0, tau_detect=10.0)
        self.assertNotIn("Z", tr.node_taint())
        aml = [FlowEdge(0, "U", "M1", A0, 0.0), FlowEdge(1, "M1", "OFFSHORE", 2_500.0, 1.0, "WIRE"),
               FlowEdge(2, "M1", "M2", 2_500.0, 2.0, "TRANSFER"), FlowEdge(3, "M2", "EXT9", 1_000.0, 3.0)]
        tr2 = TaintedFlowTracker(aml, EdgeSchema.amlsim(), kappa=0.0, min_taint=0.0,
                                 external_accounts=["EXT9"]).trace(0, tau_detect=4.0)
        ex = tr2.exited()
        self.assertAlmostEqual(ex[EDGE_EXTERNAL_WIRE], 2_500.0 + 1_000.0, places=9)
        self.assertAlmostEqual(tr2.node_taint()["M2"], 1_500.0, places=9)
        self.assertAlmostEqual(tr2.node_taint()["M1"], 5_000.0, places=9)
        sch = EdgeSchema.paysim()
        self.assertEqual(sch.classify(FlowEdge(0, "C1", "M77", 10.0, 0.0, "PAYMENT")), EDGE_MERCHANT)
        self.assertEqual(sch.classify(FlowEdge(0, "C1", "C2", 10.0, 0.0, "TRANSFER")), EDGE_INTERNAL)

    def test_window_and_causality(self) -> None:
        # Edge -1 shares tau_0 but precedes e_0 in the within-step order.
        edges = [FlowEdge(-1, "M1", "EARLY", 100.0, 0.0)] + chain_edges() + [
            FlowEdge(7, "M3", "LATE", 500.0, 12.0)]
        t = tracker(edges)
        tr = t.trace(0, tau_detect=10.0)
        self.assertNotIn("EARLY", tr.node_taint())   # precedes e_0 -> no taint
        self.assertNotIn("LATE", tr.node_taint())    # after tau_detect -> ignored
        with self.assertRaises(ValueError):
            t.trace(0, tau_detect=W_REC + 0.5)
        with self.assertRaises(ValueError):
            t.trace(0, tau_detect=-1.0)
        self.assertEqual(t.trace(0).tau_detect, W_REC)
        with self.assertRaises(ValueError):
            TaintedFlowTracker([FlowEdge(1, "a", "b", 1.0, 0.0), FlowEdge(1, "b", "c", 1.0, 1.0)])

    def test_paysim_amlsim_and_canonical_frames_agree(self) -> None:
        # PaySim customers are 'C...' (the 'M' prefix denotes merchants).
        ref = {f"C{v[1:]}": t for v, t in
               tracker(chain_edges()).trace(0, tau_detect=10.0).node_taint().items()}
        opening_c = {f"C{v[1:]}": b for v, b in OPENING.items()}
        paysim = pd.DataFrame({
            "step": [0, 1, 3, 5], "type": ["TRANSFER", "TRANSFER", "TRANSFER", "CASH_OUT"],
            "amount": [A0, 6_000.0, 4_000.0, 1_000.0], "nameOrig": ["C0", "C1", "C2", "C3"],
            "nameDest": ["C1", "C2", "C3", "C_AGENT"], "isFraud": [1, 0, 0, 0]})
        ps = TaintedFlowTracker.from_frame(paysim, "paysim", kappa=KAPPA, opening_balances=opening_c,
                                           min_taint=0.0).trace(0, tau_detect=10.0).node_taint()
        canon = pd.DataFrame({"tx_id": [0, 1, 2, 3], "event_time": [0, 3600, 3 * 3600, 5 * 3600],
                              "src": paysim["nameOrig"], "dst": paysim["nameDest"],
                              "amount": paysim["amount"], "tx_type": paysim["type"], "label": [1, 0, 0, 0]})
        cn = TaintedFlowTracker.from_frame(canon, "canonical", kappa=KAPPA, opening_balances=opening_c,
                                           min_taint=0.0).trace(0, tau_detect=10.0).node_taint()
        for other in (ps, cn):
            self.assertEqual(set(other), set(ref))
            for v in ref:
                self.assertAlmostEqual(other[v], ref[v], places=9)
        # AMLSim v1 (integer days -> 24 h) with a WIRE exit.
        aml = pd.DataFrame({"TX_ID": [10, 11, 12], "SENDER_ACCOUNT_ID": [1, 2, 3],
                            "RECEIVER_ACCOUNT_ID": [2, 3, 4], "TX_TYPE": ["TRANSFER", "TRANSFER", "WIRE"],
                            "TX_AMOUNT": [3_000.0, 3_000.0, 1_000.0], "TIMESTAMP": [0, 0, 0],
                            "IS_FRAUD": [True, False, False]})
        tr = TaintedFlowTracker.from_frame(aml, "amlsim", kappa=KAPPA, min_taint=0.0).trace(10)
        self.assertAlmostEqual(tr.node_taint()["3"], 2_000.0, places=9)
        self.assertAlmostEqual(tr.exited()[EDGE_EXTERNAL_WIRE], 1_000.0, places=9)
        self.assertEqual(TaintedFlowTracker.edges_from_amlsim(aml.assign(TIMESTAMP=[0, 1, 1]))[1].tau, 24.0)

    def test_paysim_inflow_and_opening_balances(self) -> None:
        raw = pd.DataFrame({"step": [0, 1], "type": ["CASH_IN", "TRANSFER"], "amount": [500.0, 50.0],
                            "nameOrig": ["C1", "CQ"], "nameDest": ["M999", "CR"],
                            "newbalanceOrig": [2_500.0, 10.0], "newbalanceDest": [0.0, 60.0]})
        bal = TaintedFlowTracker.paysim_opening_balances(raw, before_step=2)
        self.assertEqual(bal, {"C1": 2_500.0, "CQ": 10.0, "CR": 60.0})
        # CASH_IN credits the originator (PaySim semantics) and carries no taint.
        edges = [FlowEdge(0, "CU", "C1", 1_000.0, 0.0), FlowEdge(1, "C1", "M500", 3_000.0, 0.5, "CASH_IN"),
                 FlowEdge(2, "C1", "C2", 2_000.0, 1.0)]
        tr = TaintedFlowTracker(edges, EdgeSchema.paysim(), kappa=0.0, min_taint=0.0).trace(0, 2.0)
        self.assertAlmostEqual(tr.node_taint()["C2"], 1_000.0 * 0.5, places=9)  # pre-balance 4000
        self.assertNotIn("M500", tr.node_taint())

    def test_max_hops_and_min_taint_pruning(self) -> None:
        edges = [FlowEdge(0, "U", "H0", 1_000.0, 0.0)] + [
            FlowEdge(i, f"H{i - 1}", f"H{i}", 1_000.0, float(i)) for i in range(1, 9)]
        tr = tracker(edges, opening_balances={}, kappa=0.0, max_hops=6).trace(0, tau_detect=12.0)
        self.assertAlmostEqual(tr.beyond_horizon, 1_000.0, places=9)
        self.assertEqual(max(tr.min_hop().values()), 6)
        self.assertLess(abs(tr.conservation_residual()), TOL)
        edges2 = [FlowEdge(0, "U", "A", 1_000.0, 0.0), FlowEdge(1, "A", "B", 5.0, 1.0)]
        tr2 = tracker(edges2, opening_balances={}, kappa=0.0, min_taint=0.01).trace(0, 2.0)
        self.assertAlmostEqual(tr2.pruned, 5.0, places=9)
        self.assertNotIn("B", tr2.node_taint())
        self.assertLess(abs(tr2.conservation_residual()), TOL)

    def test_conservation_and_bounds_on_random_networks(self) -> None:
        rng = np.random.default_rng(SEED)
        for _ in range(25):
            edges, opening = random_network(rng)
            tr = TaintedFlowTracker(edges, "generic", kappa=KAPPA, opening_balances=opening,
                                    min_taint=0.0).trace(0, tau_detect=20.0)
            self.assertLess(abs(tr.conservation_residual()), 1e-7)
            self.assertTrue(all(t >= 0.0 for t in tr.node_taint().values()))
            for v, amt in tr.liquid_recoverable().items():
                self.assertLessEqual(amt, tr.balances[v] + 1e-9)
            self.assertLessEqual(tr.metrics()["recovered_amount"], 5_000.0 + 1e-9)

    def test_from_config(self) -> None:
        t = TaintedFlowTracker.from_config(chain_edges(), "paysim", HYPERPARAMS, PROJECT_CONFIG)
        self.assertEqual(t.kappa, KAPPA)
        self.assertEqual(t.window_hours, W_REC)
        self.assertEqual(t.max_hops, HYPERPARAMS["tdtpf"]["max_hops"])
        self.assertEqual(t.min_taint, HYPERPARAMS["tdtpf"]["min_taint"])


# --------------------------------------------------------------------------- #
# 2. Time decay                                                               #
# --------------------------------------------------------------------------- #
class TestTimeDecay(unittest.TestCase):

    def _single_hop(self, kappa: float, dt: float):
        edges = [FlowEdge(0, "U", "A", 1_000.0, 0.0), FlowEdge(1, "A", "B", 1_000.0, dt)]
        return tracker(edges, kappa=kappa, opening_balances={}).trace(0, tau_detect=min(dt + 0.5, W_REC))

    def test_single_hop_decay_is_exponential(self) -> None:
        for dt in (0.0, 0.5, 2.0, 6.0, 12.0, 23.0):
            tr = self._single_hop(KAPPA, dt)
            self.assertAlmostEqual(tr.node_taint()["B"], 1_000.0 * math.exp(-KAPPA * dt), places=9)
            self.assertAlmostEqual(tr.attribution_loss, 1_000.0 * (1.0 - math.exp(-KAPPA * dt)), places=9)

    def test_kappa_zero_is_pure_proportional_flow(self) -> None:
        tr = tracker(chain_edges(), kappa=0.0).trace(0, tau_detect=10.0)
        hand = chain_hand(0.0)
        self.assertEqual(tr.attribution_loss, 0.0)
        for v in ("M1", "M2", "M3"):
            self.assertAlmostEqual(tr.node_taint()[v], hand["held"][v], places=9)
        self.assertAlmostEqual(tr.held + tr.exited_total, A0, places=9)

    def test_monotone_in_kappa_and_elapsed_time(self) -> None:
        kappas = [0.0, 0.02, 0.05, 0.10, 0.25, 0.5]
        vals = [tracker(chain_edges(), kappa=k).trace(0, 10.0).node_taint()["M3"] for k in kappas]
        self.assertTrue(all(a > b for a, b in zip(vals, vals[1:])))
        dts = [0.5, 1.0, 4.0, 8.0, 16.0]
        vals_t = [self._single_hop(KAPPA, dt).node_taint()["B"] for dt in dts]
        self.assertTrue(all(a > b for a, b in zip(vals_t, vals_t[1:])))

    def test_decay_telescopes_along_full_transfer_path(self) -> None:
        """With full pass-through (share = 1) the product of hop decays is exp(-kappa (tau_n - tau_0))."""
        times = [0.0, 1.5, 4.0, 4.5, 9.0, 13.25]
        edges = [FlowEdge(0, "U", "P0", 2_000.0, times[0])] + [
            FlowEdge(i, f"P{i - 1}", f"P{i}", 2_000.0, times[i]) for i in range(1, len(times))]
        tr = tracker(edges, opening_balances={}, min_taint=0.0).trace(0, tau_detect=14.0)
        last = f"P{len(times) - 1}"
        self.assertAlmostEqual(tr.node_taint()[last], 2_000.0 * math.exp(-KAPPA * times[-1]), places=9)
        self.assertAlmostEqual(tr.node_taint()[last] + tr.attribution_loss, 2_000.0, places=9)


# --------------------------------------------------------------------------- #
# 3. Recovery simulation                                                      #
# --------------------------------------------------------------------------- #
class TestRecoverySimulation(unittest.TestCase):

    def _sim(self, edges, **policy):
        policy.setdefault("window_hours", W_REC)
        return RecoverySimulator(tracker(edges), RecoveryPolicy(**policy))

    def test_recovered_amount_and_rate_within_24h(self) -> None:
        hand = chain_hand(KAPPA)
        out = self._sim(chain_edges(), freeze_latency_hours=0.5).simulate(0, tau_detect=6.0)
        expected = sum(min(hand["held"][v], hand["balances"][v]) for v in ("M1", "M2", "M3"))
        self.assertTrue(out.within_window)
        self.assertAlmostEqual(out.frozen, expected, places=9)
        self.assertAlmostEqual(out.recovered, expected, places=9)
        self.assertAlmostEqual(out.recovery_rate_pct, 100.0 * expected / A0, places=9)
        self.assertEqual(sum(1 for a in out.actions if a.kind == ACTION_FREEZE and a.succeeded), 3)
        pol = RecoveryPolicy()
        cost = 3 * pol.freeze_cost + pol.block_cost
        self.assertAlmostEqual(out.operational_cost, cost, places=9)
        self.assertAlmostEqual(out.net_savings, expected - cost, places=9)
        self.assertAlmostEqual(out.prevented_loss_ratio_pct, out.recovery_rate_pct, places=9)
        self.assertTrue(all(a.tau <= 0.0 + W_REC for a in out.actions))

    def test_funds_moved_during_reaction_latency_are_followed(self) -> None:
        edges = chain_edges(cash_out_tau=6.25)
        sim = self._sim(edges, freeze_latency_hours=0.5)
        late = sim.simulate(0, tau_detect=6.0)        # freeze at 6.5, after the cash-out
        early = self._sim(edges, freeze_latency_hours=0.1).simulate(0, tau_detect=6.0)
        self.assertLess(late.frozen, early.frozen)
        self.assertAlmostEqual(late.trace.exited()[EDGE_CASH_OUT],
                               chain_hand(KAPPA)["t_m3"] * 0.2 * math.exp(-KAPPA * 3.25), places=9)
        self.assertAlmostEqual(early.frozen - late.frozen, late.trace.exited()[EDGE_CASH_OUT]
                               + late.trace.attribution_loss - early.trace.attribution_loss, places=9)

    def test_nothing_recovered_outside_window(self) -> None:
        sim = self._sim(chain_edges(), freeze_latency_hours=0.5)
        expired = sim.simulate(0, tau_detect=W_REC + 1.0)
        self.assertFalse(expired.within_window)
        self.assertEqual(expired.recovered, 0.0)
        self.assertEqual(expired.recovery_rate_pct, 0.0)
        late = sim.simulate(0, tau_detect=W_REC - 0.2)  # freeze would land after the deadline
        self.assertEqual(late.frozen, 0.0)
        self.assertTrue(all(not a.executed for a in late.actions if a.kind == ACTION_FREEZE))

    def test_instrument_block_and_prevented_loss_ratio(self) -> None:
        edges = chain_edges() + [FlowEdge(20, "U", "M9", 1_500.0, 3.0, "TRANSFER", 1),
                                 FlowEdge(21, "U", "M8", 2_500.0, 10.0, "TRANSFER", 1)]
        out = self._sim(edges).simulate(0, tau_detect=6.0)
        self.assertAlmostEqual(out.follow_on_exposure, 4_000.0)
        self.assertAlmostEqual(out.follow_on_blocked, 2_500.0)  # only the debit after tau_b = 6.25
        exposure = A0 + 4_000.0
        self.assertAlmostEqual(out.exposure, exposure)
        self.assertAlmostEqual(out.prevented_loss_ratio_pct,
                               100.0 * (out.recovered + 2_500.0) / exposure, places=9)
        self.assertGreater(out.prevented_loss_ratio_pct, 0.0)
        self.assertEqual([a.kind for a in out.actions][0], ACTION_BLOCK)

    def test_merchant_chargeback(self) -> None:
        edges = [FlowEdge(0, "C1", "C2", 3_000.0, 0.0, "TRANSFER", 1),
                 FlowEdge(1, "C2", "M100", 1_200.0, 1.0, "PAYMENT", 0)]
        t = TaintedFlowTracker(edges, EdgeSchema.paysim(), kappa=KAPPA, min_taint=0.0)
        out = RecoverySimulator(t, RecoveryPolicy(merchant_chargeback_rate=0.7)).simulate(0, 3.0)
        merchant = 1_200.0 * math.exp(-KAPPA * 1.0)
        self.assertAlmostEqual(out.charged_back, 0.7 * merchant, places=9)
        self.assertAlmostEqual(out.frozen, 1_800.0, places=9)
        self.assertIn(ACTION_CHARGEBACK, [a.kind for a in out.actions])

    def test_full_account_freeze_collateral(self) -> None:
        out_t = self._sim(chain_edges()).simulate(0, 6.0)
        out_f = self._sim(chain_edges(), freeze_policy="full_account", collateral_rate=0.05).simulate(0, 6.0)
        self.assertAlmostEqual(out_t.frozen, out_f.frozen, places=9)
        clean = sum(out_f.trace.balances[v] - t for v, t in out_f.trace.node_taint().items())
        self.assertAlmostEqual(out_f.collateral_frozen, clean, places=9)
        self.assertAlmostEqual(out_t.net_savings - out_f.net_savings, 0.05 * clean, places=9)

    def test_stochastic_mode_reproducible(self) -> None:
        def run():
            sim = self._sim(chain_edges(), mode="stochastic", freeze_success_rate=0.6, seed=SEED)
            return [sim.simulate(0, 6.0).frozen for _ in range(10)]
        a, b = run(), run()
        self.assertEqual(a, b)
        self.assertGreater(len(set(a)), 1)

    def test_recovery_non_increasing_in_detection_delay(self) -> None:
        edges = chain_edges() + [FlowEdge(4, "M1", "CASH1", 3_000.0, 8.0, "CASH_OUT"),
                                 FlowEdge(5, "M2", "CASH2", 3_000.0, 15.0, "CASH_OUT")]
        curve = self._sim(edges).recovery_curve(0, [0.5, 2, 4, 7.9, 9, 14, 16, 22, 23.6, 25])
        rr = curve["recovery_rate_pct"].to_numpy()
        self.assertTrue(np.all(np.diff(rr) <= 1e-9))
        self.assertEqual(rr[-1], 0.0)

    def test_simulate_many_and_aggregate(self) -> None:
        rng = np.random.default_rng(SEED)
        edges, opening = random_network(rng)
        t = TaintedFlowTracker(edges, "generic", kappa=KAPPA, opening_balances=opening)
        sim = RecoverySimulator(t)
        frame = sim.simulate_many([0, 0, 0], [1.0, 6.0, 30.0])
        agg = sim.aggregate(frame)
        self.assertEqual(len(frame), 3)
        self.assertAlmostEqual(agg["recovered_amount"], frame["recovered_amount"].sum())
        self.assertAlmostEqual(agg["recovery_rate_pct"], 100 * frame["recovered_amount"].sum() / 15_000.0)
        self.assertAlmostEqual(agg["share_within_window"], 2 / 3)
        self.assertTrue((frame["recovery_rate_pct"] <= 100.0 + 1e-9).all())
        pol = RecoveryPolicy.from_config(HYPERPARAMS, PROJECT_CONFIG)
        self.assertEqual(pol.window_hours, W_REC)
        self.assertEqual(pol.freeze_liquid_fraction, HYPERPARAMS["tdtpf"]["freeze_liquid_fraction"])
        with self.assertRaises(ValueError):
            RecoverySimulator(t, RecoveryPolicy(window_hours=12.0))


# --------------------------------------------------------------------------- #
# 4. Verification queue                                                       #
# --------------------------------------------------------------------------- #
class TestVerificationQueue(unittest.TestCase):

    def test_latency_distribution_bounds_and_sources(self) -> None:
        q = VerificationQueue(LatencyModel.from_config(HYPERPARAMS, PROJECT_CONFIG), seed=SEED)
        rng = np.random.default_rng(SEED)
        y = (rng.random(4_000) < 0.3).astype(int)
        for i, yy in enumerate(y):
            q.submit(i, 0.0, Action.APPROVE, int(yy), 100.0)
        labels = q.drain()
        lat = np.array([l.latency_days for l in labels])
        self.assertGreaterEqual(lat.min(), 1.0 - 1e-9)
        self.assertLessEqual(lat.max(), 14.0 + 1e-9)
        legit = [l for l in labels if l.y == 0]
        self.assertTrue(all(l.source == SOURCE_MATURITY and abs(l.latency_days - 14.0) < 1e-9 for l in legit))
        fraud = [l for l in labels if l.y == 1]
        share = np.mean([l.source == SOURCE_DISPUTE for l in fraud])
        self.assertLess(abs(share - 0.6), 0.05)
        disp = np.median([l.latency_days for l in fraud if l.source == SOURCE_DISPUTE])
        cb = np.median([l.latency_days for l in fraud if l.source == SOURCE_CHARGEBACK])
        self.assertLess(abs(disp - 2.0), 0.25)
        self.assertLess(abs(cb - 7.0), 0.5)

    def test_labels_held_and_released_chronologically(self) -> None:
        q = VerificationQueue(seed=SEED)
        delays_days = [5, 1, 14, 3, 3, 9, 2]
        for i, d in enumerate(delays_days):
            q.submit(f"t{i}", float(i), Action.HOLD, i % 2, 10.0 * (i + 1), delay_hours=24.0 * d)
        due = sorted((float(i) + 24.0 * d, i) for i, d in enumerate(delays_days))
        released = []
        for now in (0.0, 24.5, 50.0, 100.0, 200.0, 400.0):
            batch = q.release_until(now)
            for lab in batch:
                self.assertLessEqual(lab.tau_release, now)
                self.assertTrue(q.is_label_available(lab.tx_id, now))
            for i, d in enumerate(delays_days):
                if float(i) + 24.0 * d > now:
                    self.assertFalse(q.is_label_available(f"t{i}", now))
            released.extend(batch)
        self.assertEqual([l.tx_id for l in released], [f"t{i}" for _, i in due])
        rel = [l.tau_release for l in released]
        self.assertEqual(rel, sorted(rel))
        self.assertEqual(q.n_pending, 0)

    def test_equal_release_times_keep_submission_order(self) -> None:
        q = VerificationQueue(seed=SEED)
        for i in range(5):
            q.submit(f"x{i}", 10.0, Action.APPROVE, 0, 1.0, delay_hours=48.0)
        self.assertEqual([l.tx_id for l in q.release_until(58.0)], [f"x{i}" for i in range(5)])

    def test_clock_and_input_guards(self) -> None:
        q = VerificationQueue(seed=SEED)
        q.submit("a", 0.0, Action.APPROVE, 0, 1.0)
        q.release_until(30.0)
        with self.assertRaises(ValueError):
            q.release_until(29.0)
        with self.assertRaises(ValueError):
            q.submit("b", 10.0, Action.APPROVE, 0, 1.0)
        with self.assertRaises(ValueError):
            q.submit("a", 40.0, Action.APPROVE, 0, 1.0)
        with self.assertRaises(ValueError):
            q.submit("c", 40.0, Action.APPROVE, 0, 1.0, delay_hours=2.0)
        with self.assertRaises(ValueError):
            q.submit("d", 40.0, Action.APPROVE, 2, 1.0)
        self.assertEqual(q.n_pending, 1)

    def test_immediate_telemetry_is_separate_from_labels(self) -> None:
        q = VerificationQueue(seed=SEED)
        tel_fraud = [q.submit(f"f{i}", 0.0, Action.STEP_UP, 1, 100.0) for i in range(3_000)]
        tel_legit = [q.submit(f"l{i}", 0.0, Action.STEP_UP, 0, 100.0) for i in range(3_000)]
        tel_block = q.submit("b0", 0.0, Action.BLOCK, 1, 100.0)
        pass_rate = np.mean([t.kind == TELEMETRY_STEP_UP_PASS for t in tel_fraud])
        fail_rate = np.mean([t.kind == TELEMETRY_STEP_UP_FAIL for t in tel_legit])
        self.assertLess(abs(pass_rate - q.latency.beta_bypass), 0.02)
        self.assertLess(abs(fail_rate - q.latency.legit_step_up_fail_rate), 0.01)
        self.assertEqual(tel_block.kind, TELEMETRY_BLOCK)
        self.assertEqual(q.release_until(23.99), [])  # no label before the 1-day minimum
        self.assertEqual(len(q.telemetry_log), 6_001)

    def test_deterministic_with_seed(self) -> None:
        def run():
            q = VerificationQueue(seed=SEED)
            for i in range(200):
                q.submit(i, float(i), Action.APPROVE, i % 3 == 0, 5.0)
            return [(l.tx_id, l.tau_release, l.source) for l in q.drain()]
        self.assertEqual(run(), run())


# --------------------------------------------------------------------------- #
# 5. Closed-loop adapter                                                      #
# --------------------------------------------------------------------------- #
class TestClosedLoopAdapter(unittest.TestCase):

    def test_lambda_updated_only_on_release_with_exact_aci_step(self) -> None:
        crc = make_crc()
        adapter = ClosedLoopAdapter(crc, None, VerificationQueue(seed=SEED),
                                    AdapterConfig(gate_cycle_interval_hours=0.0))
        lam0 = crc.thresholds
        adapter.record_decision("t1", 0.0, Action.APPROVE, 5_000.0, 1, delay_hours=48.0)
        adapter.record_decision("t2", 1.0, Action.BLOCK, 80.0, 0, delay_hours=24.0)
        self.assertEqual(adapter.advance(24.5).n_released, 0)
        self.assertEqual(crc.thresholds, lam0)
        step = adapter.advance(25.0)                       # t2 released first (chronological)
        self.assertEqual([l.tx_id for l in step.released], ["t2"])
        ell2 = 150.0 / 10_000.0
        lam1 = project_ordered_thresholds([v + 0.005 * (0.02 - ell2) for v in lam0], 0.0, 1.0, 0.01)
        for a, b in zip(crc.thresholds, lam1):
            self.assertAlmostEqual(a, b, places=12)
        adapter.advance(48.0)
        ell1 = min(5_000.0 / 10_000.0, 1.0)
        lam2 = project_ordered_thresholds([v + 0.005 * (0.02 - ell1) for v in lam1], 0.0, 1.0, 0.01)
        for a, b in zip(crc.thresholds, lam2):
            self.assertAlmostEqual(a, b, places=12)
        self.assertTrue(all(b < a for a, b in zip(lam0, crc.thresholds)))  # missed fraud -> more protective
        self.assertEqual(len(adapter.threshold_log), 2)
        self.assertEqual(adapter.threshold_log[1].lambda_before, tuple(lam1))
        self.assertLess(crc.guarantee()["identity_residual"], 1e-12)

    def test_martingale_wealth_updated_on_release(self) -> None:
        gate = MartingaleEProcessGate.from_config(HYPERPARAMS)
        gate.inject_active(make_rule("ACT", 0.9))
        gate.submit_candidate(make_rule("CAND", 0.9), 632)
        self.assertEqual(gate.status("CAND"), "quarantined")
        adapter = ClosedLoopAdapter(make_crc(), gate, VerificationQueue(seed=SEED),
                                    AdapterConfig(gate_cycle_interval_hours=0.0))
        ys = [1, 0, 1, 1, 0]
        for i, y in enumerate(ys):
            adapter.record_decision(f"r{i}", float(i), Action.HOLD, 200.0, y, rule_ids=("ACT", "CAND"),
                                    delay_hours=24.0 * (1 + i))
        self.assertEqual(adapter.wealth("ACT"), 1.0)
        self.assertEqual(adapter.wealth("CAND"), 1.0)
        adapter.advance(24.0 * 3 + 2.0)  # releases r0, r1, r2
        hits, fa = 2, 1
        cfg = GATE_CFG
        roll = (cfg.q1 / cfg.q0) ** fa * ((1 - cfg.q1) / (1 - cfg.q0)) ** hits
        admit = (cfg.p1 / cfg.p0) ** hits * ((1 - cfg.p1) / (1 - cfg.p0)) ** fa
        self.assertAlmostEqual(adapter.wealth("ACT"), roll, places=12)
        self.assertAlmostEqual(adapter.wealth("CAND"), admit, places=12)
        self.assertEqual(len(adapter.gate_reports), 1)
        adapter.drain()
        hits, fa = 3, 2
        self.assertAlmostEqual(adapter.wealth("ACT"),
                               (cfg.q1 / cfg.q0) ** fa * ((1 - cfg.q1) / (1 - cfg.q0)) ** hits, places=12)
        self.assertEqual(adapter.n_rule_outcomes_applied, 10)

    def test_degraded_rule_rolled_back_and_retraining_scheduled(self) -> None:
        gate = MartingaleEProcessGate.from_config(HYPERPARAMS)
        gate.inject_active(make_rule("BAD", 0.9))
        fired = []
        adapter = ClosedLoopAdapter(make_crc(), gate, VerificationQueue(seed=SEED),
                                    AdapterConfig(gate_cycle_interval_hours=6.0),
                                    on_retrain=fired.append)
        rng = np.random.default_rng(SEED)
        n_cycles = None
        for h in range(0, 24 * 20):
            for k in range(2):
                y = int(rng.random() < 0.10)  # precision collapsed to 10 %
                adapter.record_decision(f"b{h}_{k}", float(h), Action.BLOCK, 300.0, y,
                                        rule_ids=("BAD",), delay_hours=24.0)
            step = adapter.advance(float(h) + 0.5)
            if step.gate_report is not None and "BAD" in step.gate_report.rolled_back:
                n_cycles = len(adapter.gate_reports)
                break
        self.assertIsNotNone(n_cycles)
        self.assertLessEqual(n_cycles, GATE_CFG.max_rollback_cycles)
        self.assertEqual(gate.status("BAD"), "quarantined")
        self.assertTrue(any(REASON_ROLLBACK in t.reasons and "BAD" in t.rolled_back_rules for t in fired))
        self.assertEqual(adapter.summary()["n_rollbacks"], 1)

    def test_gate_cycles_respect_interval(self) -> None:
        gate = MartingaleEProcessGate.from_config(HYPERPARAMS)
        gate.inject_active(make_rule("R", 0.9))
        adapter = ClosedLoopAdapter(make_crc(), gate, VerificationQueue(seed=SEED),
                                    AdapterConfig(gate_cycle_interval_hours=6.0))
        for h in range(48):
            adapter.record_decision(f"g{h}", float(h), Action.HOLD, 50.0, 1, rule_ids=("R",),
                                    delay_hours=24.0)
        for h in range(24, 72):
            adapter.advance(float(h))
        times = [24.0 + 6.0 * k for k in range(len(adapter.gate_reports))]
        self.assertEqual(len(adapter.gate_reports), 8)
        self.assertEqual([r.n_observations for r in adapter.gate_reports], [1] + [6] * 7)
        self.assertEqual(adapter._last_gate_cycle, times[-1])
        adapter.drain()
        self.assertEqual(adapter.n_rule_outcomes_applied, 48)

    def test_retrain_volume_and_drift_triggers_with_cooldown(self) -> None:
        cfg = AdapterConfig(gate_cycle_interval_hours=0.0, retrain_min_labels=50, retrain_min_frauds=5,
                            retrain_cooldown_hours=24.0, drift_tolerance=0.01, drift_min_samples=20)
        adapter = ClosedLoopAdapter(make_crc(), None, VerificationQueue(seed=SEED), cfg)
        for i in range(120):
            adapter.record_decision(f"v{i}", i * 0.1, Action.HOLD, 10.0, int(i % 10 == 0), delay_hours=24.0)
        adapter.advance(24.0 + 4.95)
        self.assertEqual(len(adapter.retrain_schedule), 1)
        self.assertEqual(adapter.retrain_schedule[0].reasons, (REASON_VOLUME,))
        adapter.advance(24.0 + 11.95)
        self.assertEqual(len(adapter.retrain_schedule), 1)  # cooling down
        adapter.advance(60.0)
        self.assertEqual(len(adapter.retrain_schedule), 2)
        # Drift: every approved fraud costs ell = 1 >> alpha.
        drift = ClosedLoopAdapter(make_crc(), None, VerificationQueue(seed=SEED), cfg)
        for i in range(30):
            drift.record_decision(f"d{i}", float(i), Action.APPROVE, 20_000.0, 1, delay_hours=24.0)
        drift.advance(24.0 + 29.0)
        self.assertTrue(drift.retrain_schedule and REASON_DRIFT in drift.retrain_schedule[0].reasons)

    def test_end_to_end_delayed_stream(self) -> None:
        rng = np.random.default_rng(SEED)
        crc = make_crc()
        gate = MartingaleEProcessGate.from_config(HYPERPARAMS)
        gate.submit_candidate(make_rule("GOOD", 0.95), 632)
        confirmed = []
        adapter = ClosedLoopAdapter(crc, gate, VerificationQueue(seed=SEED),
                                    AdapterConfig(gate_cycle_interval_hours=6.0),
                                    on_fraud_confirmed=confirmed.append)
        n, n_fraud = 0, 0
        for h in range(24 * 21):
            for _ in range(4):
                y = int(rng.random() < 0.05)
                score = float(np.clip(rng.normal(0.75 if y else 0.15, 0.15), 0.0, 1.0))
                amount = float(rng.lognormal(6.0, 1.0))
                fires = ("GOOD",) if (y and rng.random() < 0.8) or (not y and rng.random() < 0.002) else ()
                action, tel = adapter.route_and_record(f"s{n}", float(h), score, amount, y, fires)
                self.assertEqual(int(action), tel.action)
                n += 1
                n_fraud += y
            step = adapter.advance(float(h) + 0.99)
            for lab in step.released:
                self.assertLessEqual(lab.tau_release, float(h) + 0.99)
                self.assertGreaterEqual(lab.latency_days, 1.0 - 1e-9)
        adapter.drain()
        self.assertEqual(adapter.n_labels_applied, n)
        self.assertEqual(len(confirmed), n_fraud)
        self.assertEqual(crc.t, n)
        g = crc.guarantee()
        self.assertTrue(g["within_bound"])
        self.assertLess(g["identity_residual"], 1e-9)
        self.assertTrue(crc.is_strictly_monotone())
        self.assertEqual(gate.status("GOOD"), "active")
        self.assertGreater(adapter.summary()["n_promotions"], 0)
        frame = adapter.threshold_frame()
        self.assertEqual(len(frame), n)
        self.assertTrue((frame["tau_release"].diff().dropna() >= 0).all())

    def test_from_config_and_deterministic_replay(self) -> None:
        a = ClosedLoopAdapter.from_config(HYPERPARAMS, PROJECT_CONFIG)
        self.assertEqual(a.config.gate_cycle_interval_hours,
                         HYPERPARAMS["closed_loop"]["rollback_check_interval_steps"])
        self.assertEqual(a.queue.latency.max_days, HYPERPARAMS["closed_loop"]["label_delay_days"][1])
        self.assertEqual(a.crc.alpha, HYPERPARAMS["conformal_crc"]["target_risk_alpha"])

        def run():
            rng = np.random.default_rng(SEED)
            ad = ClosedLoopAdapter(make_crc(), None, VerificationQueue(seed=SEED))
            for i in range(300):
                y = int(rng.random() < 0.1)
                ad.route_and_record(i, i * 0.5, float(rng.random()), 100.0, y)
                ad.advance(i * 0.5)
            ad.drain()
            return ad.crc.thresholds, len(ad.threshold_log)
        self.assertEqual(run(), run())


if __name__ == "__main__":
    unittest.main(verbosity=2)
