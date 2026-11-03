"""
src/baselines
=============

Re-implemented state-of-the-art comparison baselines (Phase 6) sharing the
unified ``fit / predict_proba / predict / measure_latency`` contract defined
in :mod:`src.baselines.base`.

=====  ==========================  ==============================================
Key    Class                       Reference
=====  ==========================  ==============================================
bp1    :class:`EARNRXTJ`           Almazroi & Ayub (2023), IEEE Access
bp2    :class:`LambdaGNNAutoencoder` Alarfaj & Shahzadi (2025), IEEE Access
bp3    :class:`CAREGNN`            Dou et al. (2020), ACM CIKM
bp4    :class:`ARGUSConformal`     Nayak & Bushara (2026), IEEE Access
=====  ==========================  ==============================================
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Type

from .argus_conformal import ARGUSConfig, ARGUSConformal, crc_fnr_threshold
from .base import SEED, BaselineModel, seed_everything
from .care_gnn import CAREGNN, CAREGNNConfig
from .earn_rxt_j import EARNRXTJ, EARNRXTJConfig, jaya_minimize, smote_oversample
from .lambda_gnn_ae import LambdaGNNAEConfig, LambdaGNNAutoencoder

BASELINE_REGISTRY: Dict[str, Type[BaselineModel]] = {
    "bp1": EARNRXTJ,
    "bp2": LambdaGNNAutoencoder,
    "bp3": CAREGNN,
    "bp4": ARGUSConformal,
}

_ALIASES: Dict[str, str] = {
    "earn_rxt_j": "bp1", "earn+rxt-j": "bp1", "earn-rxt-j": "bp1", "almazroi": "bp1",
    EARNRXTJ.name: "bp1",
    "lambda_gnn_ae": "bp2", "gnn_ae": "bp2", "lambda": "bp2", "alarfaj": "bp2",
    LambdaGNNAutoencoder.name: "bp2",
    "care_gnn": "bp3", "care-gnn": "bp3", "caregnn": "bp3", "dou": "bp3",
    CAREGNN.name: "bp3",
    "argus": "bp4", "argus_conformal": "bp4", "nayak": "bp4",
    ARGUSConformal.name: "bp4",
}


def resolve_baseline_name(name: str) -> str:
    """Canonical registry key (``bp1`` ... ``bp4``) for a name or alias."""
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in BASELINE_REGISTRY:
        raise KeyError(f"Unknown baseline '{name}'. Known: {sorted(BASELINE_REGISTRY)} "
                       f"and aliases {sorted(_ALIASES)}")
    return key


def get_baseline_model(name: str, config: Optional[Mapping[str, Any]] = None,
                       seed: int = SEED) -> BaselineModel:
    """Factory: ``get_baseline_model("bp3", {"epochs": 10})`` -> unfitted model.

    ``config`` may be ``None`` (published defaults), a plain mapping of the
    baseline's config fields, the baseline's config dataclass, or a mapping
    keyed by the canonical name (``{"bp3": {...}, "bp4": {...}}``) - in the
    last case only this baseline's entry is used, so one experiment file can
    configure all four models.
    """
    key = resolve_baseline_name(name)
    cls = BASELINE_REGISTRY[key]
    if isinstance(config, Mapping) and any(k in BASELINE_REGISTRY for k in config):
        config = config.get(key)
    return cls(config=config, seed=seed)


__all__ = [
    "ARGUSConfig", "ARGUSConformal", "BASELINE_REGISTRY", "BaselineModel", "CAREGNN",
    "CAREGNNConfig", "EARNRXTJ", "EARNRXTJConfig", "LambdaGNNAEConfig", "LambdaGNNAutoencoder",
    "SEED", "crc_fnr_threshold", "get_baseline_model", "jaya_minimize", "resolve_baseline_name",
    "seed_everything", "smote_oversample",
]
