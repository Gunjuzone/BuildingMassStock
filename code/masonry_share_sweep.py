"""Sweep the masonry share assumed for the buildings with no recorded construction year.

Usage: python code/masonry_share_sweep.py --tag baseline_v3
"""
from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline import conditional, P_MASONRY_BY_PERIOD
from stock_engine import StockModel, load_buildings, load_rasmi_ranges, run_monte_carlo

ROOT = Path(__file__).resolve().parents[1]
BUILDINGS = Path(os.environ.get("BUILDINGS",
                               ROOT / "data" / "buildings" / "study_buildings_enriched.csv"))
SHARES = sorted({round(0.50 + 0.02 * i, 2) for i in range(21)} | {0.65, 0.72, 0.79, 0.85})


def main(tag: str, n_iter: int, seed: int):
    b = load_buildings(BUILDINGS)
    q = load_rasmi_ranges()
    undated = int(b["period"].isna().sum())
    base = replace(conditional(), p_masonry_by_period=P_MASONRY_BY_PERIOD)

    rows = []
    for s in SHARES:
        cfg = replace(base, p_masonry={"RM": s, "NR": s})
        totals, _ = run_monte_carlo(StockModel(b, q, cfg), n_iter, seed, keep_buildings=False)
        m = totals.mean(axis=0)
        rows.append({"undated_masonry_share": s, "concrete_Mkg": m[0], "brick_Mkg": m[1],
                     "brick_minus_concrete_Mkg": m[1] - m[0]})
    d = pd.DataFrame(rows)

    below = d[d.brick_minus_concrete_Mkg < 0].undated_masonry_share.max()
    above = d[d.brick_minus_concrete_Mkg > 0].undated_masonry_share.min()
    out = ROOT / "analysis" / "results" / tag / "E16_masonry_share_sweep.csv"
    d.to_csv(out, index=False)
    print(d.round(1).to_string(index=False))
    print(f"\n{undated} buildings have no recorded year and take this share.")
    print(f"brick overtakes concrete between {below} and {above}")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline_v3")
    ap.add_argument("--n-iter", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.tag, a.n_iter, a.seed)
