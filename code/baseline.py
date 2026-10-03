"""The one definition of the model the manuscript describes."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from stock_engine import CADASTRAL_STOREY_IQR, Config, ErrorComponents as EC, ROOT

MEASURED_HEIGHT = EC(0.40, 0.0, 0.11)
MEASURED_FOOTPRINT = EC(0.33, 0.0, 0.06)

P_MASONRY_BY_PERIOD = {"<1950": 0.98, "1950-1962": 0.86, "1963-1968": 0.57,
                       "1969-1974": 0.44, "1975-1994": 0.28, ">1994": 0.10}
P_MASONRY_UNDATED = {"RM": 0.79, "NR": 0.79}


def label_flip_rates() -> dict:
    path = ROOT / "analysis" / "label_error_rates.json"
    if not path.exists():
        raise FileNotFoundError("run code/build_buildings_table.py first")
    return {k: v["flip_probability"] for k, v in json.loads(path.read_text())["rates"].items()}


def submitted() -> Config:
    """The design as submitted: 9 m height threshold, flat masonry share, no label error, assumed 0.15."""
    return Config(mi_mode="p50", block_size_m=500.0, typology_source="label")


def conditional() -> Config:
    """The corrected model with material intensity held at its median, so the effect of sampling MI can be read against it."""
    return replace(submitted(), structure_rule="period", p_masonry_by_period=P_MASONRY_BY_PERIOD,
                   p_masonry=P_MASONRY_UNDATED, label_flip=label_flip_rates(),
                   height=MEASURED_HEIGHT, footprint=MEASURED_FOOTPRINT)


def baseline() -> Config:
    """The reference run the headline figures come from: everything propagated, MI half shared."""
    return replace(conditional(), mi_mode="stochastic", mi_w_category=0.5)


def classifier_probs() -> Config:
    """Typology sampled from the classifier's probabilities instead of fixed to the zoning label."""
    return replace(conditional(), typology_source="probabilities")


def with_storey_height_range(cfg: Config) -> Config:
    """The decomposition gives the storey height the cadastral interquartile range."""
    return replace(cfg, floor_height_range=CADASTRAL_STOREY_IQR)
