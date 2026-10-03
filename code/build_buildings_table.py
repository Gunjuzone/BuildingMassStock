"""Build the enriched buildings table used by stock_engine.py, and measure label-error rates.

Usage: python code/build_buildings_table.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from stock_engine import CLASS_TO_FUNCTION, FUNCTIONS, PERIOD_EDGES, PERIOD_LABELS, ROOT

OUT_CSV = ROOT / "data" / "buildings" / "study_buildings_enriched.csv"
OUT_JSON = ROOT / "analysis" / "label_error_rates.json"
MIN_IOU = 0.5
MAX_MATCH_DIST_M = 10.0


def main(min_iou: float = MIN_IOU):
    b = pd.read_csv(ROOT / "data" / "buildings" / "study_buildings.csv")
    m = pd.read_csv(ROOT / "data" / "validation" / "cadastre" / "study_buildings_cadastre_match.csv")
    if not (np.allclose(b["X_coord"], m["X_coord"]) and np.allclose(b["Y_coord"], m["Y_coord"])):
        raise ValueError("building table and cadastre match are not row-aligned")

    b["iou"] = m["iou"].to_numpy()
    b["match_dist_m"] = m["match_dist_m"].to_numpy()
    b["footprint_ratio"] = (b["Area"] / m["cad_footprint_m2"]).to_numpy()
    near = b["iou"] > min_iou
    b["match_ok"] = near

    b["cad_current_use"] = m["cad_current_use"].where(near)
    b["year_built"] = np.where(near, m["cad_year_built"], np.nan)
    b["period"] = pd.cut(b["year_built"], PERIOD_EDGES, labels=PERIOD_LABELS).astype(object)
    floors = m["cad_max_floors_above"].to_numpy(float)
    b["recorded_floors"] = np.where(b["match_ok"] & (floors > 0), floors, np.nan)
    b["cad_official_gfa_m2"] = np.where(near, m["cad_official_gfa_m2"], np.nan)

    ag = ROOT / "data" / "classifier" / "ag_zoning_morphology" / "oof_predictions.csv"
    probs = [f"Prob_{c}" for c in ("Residential", "Mixed-Use", "Institutional", "Amenities")]
    o = pd.read_csv(ag)[["Polygon_ID", "Predicted", *probs]]
    before = b[probs].max(axis=1).mean()
    b = b.drop(columns=[c for c in [*probs, "Predicted"] if c in b]).merge(o, on="Polygon_ID", how="left")
    if b[probs].isna().any().any():
        raise ValueError("some buildings have no out-of-fold prediction")
    print(f"  classifier probabilities from {ag.parent.name}: "
          f"mean top-class {before:.3f} -> {b[probs].max(axis=1).mean():.3f}")

    b.to_csv(OUT_CSV, index=False)

    f = b[near & b["cad_current_use"].notna()].copy()
    f["zoning_function"] = f["Typology"].map(CLASS_TO_FUNCTION)
    f["cadastre_function"] = np.where(f["cad_current_use"] == "1_residential", "RM", "NR")
    table = pd.crosstab(f["zoning_function"], f["cadastre_function"])
    rates = {}
    for fn in FUNCTIONS:
        n = int(table.loc[fn].sum())
        wrong = int(table.loc[fn].drop(fn).sum())
        rates[fn] = {"n": n, "disagree": wrong, "flip_probability": wrong / n if n else 0.0}
    payload = {
        "description": "P(cadastral function differs from the zoning-derived function), by assigned function",
        "match_rule": {"rule": "footprint overlap", "min_iou": min_iou},
        "crosstab": table.to_dict(),
        "rates": rates,
        "agreement": float((f["zoning_function"] == f["cadastre_function"]).mean()),
        "n_matched": int(len(f)),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    print(f"wrote {OUT_CSV} ({len(b)} buildings)")
    print(f"  matched by footprint overlap at IoU > {min_iou:g}: {int(near.sum())}")
    sel = b[near]
    print(f"  centroid distance of those matches: median {sel.match_dist_m.median():.2f} m, "
          f"90th percentile {sel.match_dist_m.quantile(0.9):.2f} m")
    print(f"  usable for the model-form test (match_ok and floors > 0): {int(b['recorded_floors'].notna().sum())}")
    print(f"  with a construction period: {int(b['period'].notna().sum())}")
    print(b["period"].value_counts(dropna=False).to_string())
    print("\nlabel-error rates:", json.dumps(rates, indent=2))
    print(table.to_string())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--iou", type=float, default=MIN_IOU, help="minimum intersection over union")
    main(ap.parse_args().iou)
