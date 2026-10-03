"""Convergence of the Monte Carlo and of the Sobol indices."""
import argparse
import os, json
from pathlib import Path
from dataclasses import replace
import pandas as pd

from stock_engine import ROOT, StockModel, load_buildings, load_rasmi_ranges, run_monte_carlo, sobol_grouped
from baseline import conditional, with_storey_height_range

BUILDINGS = Path(os.environ.get("BUILDINGS",
                               ROOT / "data" / "buildings" / "study_buildings_enriched.csv"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="corrected_v1")
    args = ap.parse_args()
    b = load_buildings(BUILDINGS)
    q = load_rasmi_ranges()
    base = with_storey_height_range(conditional())
    full = replace(base, mi_mode="stochastic", mi_w_category=0.5)

    rows = []
    for name, cfg in [("conditional_p50", base), ("full_mi_shared_0.5", full)]:
        for n in [250, 500, 1000, 2000, 3000, 5000]:
            t, _ = run_monte_carlo(StockModel(b, q, cfg), n, seed=42, keep_buildings=False)
            m, s = t.mean(0), t.std(0, ddof=1)
            rows += [{"config": name, "n_iter": n, "material": mat,
                      "district_mean_Mkg": m[i] / 1e6, "district_cv_pct": 100 * s[i] / m[i]}
                     for i, mat in enumerate(["Concrete","Brick","Wood","Steel","Glass","Plastics","Aluminium","Copper"])]
            print(f"  MC {name} n={n}: done")
    mc = pd.DataFrame(rows)
    outdir = ROOT / "analysis" / "results" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    mc.to_csv(outdir / "E14_convergence_mc.csv", index=False)

    srows = []
    for n in [500, 1000, 2000, 4000]:
        d, _, _ = sobol_grouped(StockModel(b, q, base), n_base=n, seed=42, n_boot=50)
        srows.append(d.assign(n_base=n))
        print(f"  Sobol n_base={n}: done")
    sob = pd.concat(srows, ignore_index=True)
    sob.to_csv(outdir / "E14_convergence_sobol.csv", index=False)

    print("\nMonte Carlo, district CV (%) vs iterations:")
    print(mc[mc.material.isin(["Concrete","Brick"])].pivot_table(index="n_iter", columns=["config","material"], values="district_cv_pct").round(3).to_string())
    print("\nSobol total-order indices vs base samples (conditional_p50, Concrete):")
    print(sob[(sob.material=="Concrete")].pivot_table(index="group", columns="n_base", values="ST").round(3).to_string())

if __name__ == "__main__":
    main()
