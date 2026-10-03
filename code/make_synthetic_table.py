"""Write a synthetic buildings table, and the label-error rates that go with it.

The Maxar Precision3D vectors behind the real table are licensed and cannot be redistributed, so this
writes a table of the same shape with values drawn from simple distributions. It reproduces no result and
no real building; it exists so the Monte Carlo and the Sobol decomposition can be run end to end.

Usage: python code/make_synthetic_table.py [--n 400] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "buildings" / "study_buildings_synthetic.csv"
RATES = ROOT / "analysis" / "label_error_rates.json"

TYPOLOGIES = ["Mixed-Use", "Amenities", "Institutional", "Residential"]
TYPOLOGY_P = [0.62, 0.14, 0.12, 0.12]
PERIODS = ["<1950", "1950-1962", "1963-1968", "1969-1974", "1975-1994", ">1994"]
PERIOD_P = [0.33, 0.19, 0.07, 0.10, 0.14, 0.17]
USES = ["residential", "public services", "office", "agriculture", "industrial"]
USE_P = [0.78, 0.09, 0.05, 0.06, 0.02]


def main(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    area = np.round(np.exp(rng.normal(5.0, 0.8, n)), 2)
    height = np.round(np.exp(rng.normal(1.95, 0.55, n)), 2)
    side = np.sqrt(area)
    typ = rng.choice(TYPOLOGIES, n, p=TYPOLOGY_P)
    dated = rng.random(n) < 0.45
    period = np.where(dated, rng.choice(PERIODS, n, p=PERIOD_P), None)
    year = {"<1950": 1935, "1950-1962": 1956, "1963-1968": 1965,
            "1969-1974": 1971, "1975-1994": 1985, ">1994": 2004}
    d = pd.DataFrame({
        "Polygon_ID": np.arange(900001, 900001 + n),
        "Maxar_Group_ID": np.arange(1, n + 1),
        "Zoning_ID": rng.integers(1, 1 + max(1, n // 5), n),
        "X_coord": np.round(425000 + rng.random(n) * 5000, 3),
        "Y_coord": np.round(4580000 + rng.random(n) * 5000, 3),
        "DISTRICTE": rng.choice([5, 7, 6], n, p=[0.59, 0.25, 0.16]),
        "Area": area,
        "Perimeter": np.round(side * rng.uniform(3.6, 4.6, n), 2),
        "MBG_Width": np.round(side * rng.uniform(0.6, 1.0, n), 2),
        "MBG_Length": np.round(side * rng.uniform(1.0, 1.8, n), 2),
        "Aspect_Ratio": np.round(rng.uniform(1.0, 3.0, n), 3),
        "Height": height,
        "AMSL": np.round(rng.uniform(10, 250, n), 2),
        "GLH": np.round(rng.uniform(5, 240, n), 2),
        "Typology": typ,
        "Match_Score": rng.integers(0, 12, n),
        "Confidence_Score": np.round(rng.uniform(0, 1, n), 3),
        "Convexity": np.round(rng.uniform(0.7, 1.0, n), 3),
        "Solidity": np.round(rng.uniform(0.7, 1.0, n), 3),
        "Elongation": np.round(rng.uniform(0.2, 1.0, n), 3),
        "Rectangularity": np.round(rng.uniform(0.5, 1.0, n), 3),
        "Form_ratio": np.round(rng.uniform(0.2, 1.2, n), 3),
        "cv_fold": rng.integers(0, 5, n),
        "iou": np.round(rng.uniform(0.0, 0.9, n), 3),
        "match_dist_m": np.round(rng.uniform(0, 12, n), 2),
        "footprint_ratio": np.round(rng.uniform(0.4, 2.2, n), 3),
        "cad_current_use": rng.choice(USES, n, p=USE_P),
        "period": period,
        "recorded_floors": np.where(dated, np.maximum(1, np.round(height / 3.0)), np.nan),
        "cad_official_gfa_m2": np.where(dated, np.round(area * np.maximum(1, np.round(height / 3.0)), 1), np.nan),
    })
    d["year_built"] = [year[p] if p else np.nan for p in d["period"]]
    d["match_ok"] = d["iou"] > 0.5
    for t in TYPOLOGIES:
        d[f"Prob_{t}"] = 0.0
    p = rng.dirichlet(np.ones(len(TYPOLOGIES)) * 1.5, n)
    for i, t in enumerate(TYPOLOGIES):
        d[f"Prob_{t}"] = np.round(p[:, i], 4)
    d["Predicted"] = [TYPOLOGIES[i] for i in p.argmax(axis=1)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(d)} rows, {len(d.columns)} columns)")

    if not RATES.exists():
        matched = d[d["match_ok"]]
        rm = matched[matched["Typology"].isin(["Residential", "Mixed-Use"])]
        nr = matched[~matched["Typology"].isin(["Residential", "Mixed-Use"])]
        rates = {}
        for name, part in (("RM", rm), ("NR", nr)):
            disagree = int((part["cad_current_use"] != "residential").sum()) if name == "RM" \
                else int((part["cad_current_use"] == "residential").sum())
            n = max(1, len(part))
            rates[name] = {"n": n, "disagree": disagree, "flip_probability": disagree / n}
        RATES.parent.mkdir(parents=True, exist_ok=True)
        RATES.write_text(json.dumps({"description": "synthetic label-error rates", "rates": rates}, indent=2))
        print(f"wrote {RATES}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.n, a.seed)
