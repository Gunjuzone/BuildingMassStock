"""Fit the input error model to the cadastral records.

Usage: python code/fit_error_model.py [--iou 0.5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MATCH = ROOT / "data" / "validation" / "cadastre" / "study_buildings_cadastre_match.csv"
OUT = ROOT / "analysis" / "error_model_fitted.json"
SAMPLE = ROOT / "analysis" / "cadastre_gfa_error_sample.csv"
BLOCK_M = 500.0


def residuals(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()
    m["floors_h3"] = np.maximum(1, np.round(m.Height / 3.0))
    m["model_gfa"] = m.Area * m.floors_h3
    m = m[(m.model_gfa > 0) & (m.cad_parts_gfa_above_m2 > 0) & (m.cad_footprint_m2 > 0)
          & (m.cad_max_floors_above > 0)]
    m["r_gfa"] = np.log(m.model_gfa / m.cad_parts_gfa_above_m2)
    m["r_foot"] = np.log(m.Area / m.cad_footprint_m2)
    m["r_height"] = np.log(m.Height / (3.0 * m.cad_max_floors_above))
    return m


def block_share(m: pd.DataFrame, col: str) -> float:
    """Share of residual variance common to buildings in the same 500 m block."""
    blk = ((m.X_coord // BLOCK_M).astype(int).astype(str) + "_"
           + (m.Y_coord // BLOCK_M).astype(int).astype(str))
    g = m.groupby(blk)[col]
    n = g.size()
    keep = n[n >= 2].index
    sel = m[blk.isin(keep)]
    if len(sel) < 30:
        return float("nan")
    b = blk[blk.isin(keep)]
    gb = sel.groupby(b)[col]
    ni = gb.size().to_numpy(float)
    N, k = ni.sum(), len(ni)
    if k < 2:
        return float("nan")
    grand = sel[col].mean()
    msb = (ni * (gb.mean().to_numpy() - grand) ** 2).sum() / (k - 1)
    msw = ((gb.var(ddof=1).to_numpy() * (ni - 1)).sum()) / (N - k)
    n0 = (N - (ni ** 2).sum() / N) / (k - 1)
    s2_between = max(0.0, (msb - msw) / n0)
    return float(np.clip(s2_between / (s2_between + msw), 0, 1))


def fit(m: pd.DataFrame) -> dict:
    return {
        "n": int(len(m)),
        "sigma_height": round(float(m.r_height.std()), 4),
        "sigma_footprint": round(float(m.r_foot.std()), 4),
        "sigma_gfa": round(float(m.r_gfa.std()), 4),
        "median_gfa_ratio": round(float(np.exp(m.r_gfa.median())), 4),
        "w_block_height": round(block_share(m, "r_height"), 4),
        "w_block_footprint": round(block_share(m, "r_foot"), 4),
        "w_block_gfa": round(block_share(m, "r_gfa"), 4),
    }


MAX_D = 5.0
MIN_IOU_SANITY = 0.3


def select(raw: pd.DataFrame) -> pd.DataFrame:
    """Position-based selection, independent of the geometry being measured."""
    return raw[(raw.match_dist_m <= MAX_D) & (raw.iou > MIN_IOU_SANITY)]


def main(iou: float):
    raw = pd.read_csv(MATCH)
    print("  IoU >      n   sig_H  sig_A  sig_GFA   w_blk_H  w_blk_A")
    curve = {}
    for t in np.arange(0.10, 0.85, 0.05):
        m = residuals(raw[raw.iou > t])
        if len(m) < 50:
            continue
        f = fit(m)
        curve[round(float(t), 2)] = f
        print(f"   {t:.2f}  {f['n']:5d}   {f['sigma_height']:.3f}  {f['sigma_footprint']:.3f}  "
              f"{f['sigma_gfa']:.3f}    {f['w_block_height']:.3f}    {f['w_block_footprint']:.3f}")

    sel = residuals(select(raw))
    keep = ["ID", "X_coord", "Y_coord", "DISTRICTE", "Area", "Height", "Typology",
            "cad_cadastral_ref", "cad_year_built", "cad_current_use", "cad_footprint_m2",
            "cad_parts_gfa_above_m2", "cad_max_floors_above", "match_dist_m", "iou",
            "floors_h3", "model_gfa", "r_gfa", "r_foot", "r_height"]
    out = sel[[c for c in keep if c in sel.columns]].copy()
    out["r_gfa_above"] = out.r_gfa
    out.to_csv(SAMPLE, index=False)
    print(f"wrote {SAMPLE} with {len(out)} selected pairs")

    chosen = fit(sel)
    payload = {"rule": "position, with a minimal overlap check",
               "max_centroid_distance_m": MAX_D,
               "min_iou_sanity": MIN_IOU_SANITY, "block_m": BLOCK_M,
               "fitted": chosen, "across_iou_thresholds": curve}
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"\nat IoU > {iou}: {json.dumps(chosen, indent=1)}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iou", type=float, default=0.5)
    main(ap.parse_args().iou)
