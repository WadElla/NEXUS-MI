"""Access to the version-controlled NEXUS-MI study protocol."""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import yaml


@lru_cache(maxsize=1)
def paper_protocol() -> dict:
    """Load the study protocol bundled with the NEXUS-MI package."""
    resource = files("nexus_mi").joinpath("paper_protocol.yaml")
    try:
        text = resource.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError("Bundled NEXUS-MI study protocol was not found.") from exc
    cfg = yaml.safe_load(text)
    if not isinstance(cfg, dict):
        raise RuntimeError("Invalid bundled NEXUS-MI study protocol configuration.")
    return cfg


def sensitivity_profiles() -> list[dict]:
    """Return the study availability profiles as experiment-driver arguments."""
    profiles = paper_protocol()["communication"]["sensitivity_profiles"]
    rows = []
    for name in ("mild", "default", "severe"):
        p = profiles[name]
        rows.append(
            {
                "name": name,
                "online_prob_good": float(p["high"]),
                "online_prob_med": float(p["moderate"]),
                "online_prob_bad": float(p["low"]),
            }
        )
    return rows


def reliability_thresholds_pct() -> dict[str, float]:
    """Return dataset-display-name analytical reliability thresholds."""
    src = paper_protocol()["analysis"]["reliability_threshold_pct"]
    return {"BCICIV-2a": float(src["bciciv2a"]), "OpenBMI": float(src["openbmi"])}
