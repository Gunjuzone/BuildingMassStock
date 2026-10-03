"""District stock by zoning class, with and without the measured label error."""
import argparse
import os, json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from baseline import conditional
from stock_engine import (CLASSES, MATERIALS, ROOT, Config, StockModel, load_buildings,
                          load_rasmi_ranges, run_monte_carlo)

BUILDINGS = Path(os.environ.get("BUILDINGS",
                               ROOT / "data" / "buildings" / "study_buildings_enriched.csv"))



def totals_by_class(model, per_b, assigned):
    """Mean stock per zoning class (Mkg), summing buildings by their assigned label."""
    out = {}
    for i, c in enumerate(CLASSES):
        idx = np.where(assigned == i)[0]
        out[c] = per_b[:, idx, :].sum(axis=1, dtype=np.float64).mean(axis=0) / 1e6
    return pd.DataFrame(out, index=MATERIALS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="corrected_v1")
    ap.add_argument("--n-iter", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    buildings = load_buildings(BUILDINGS)
    q = load_rasmi_ranges()
    flips = {k: v["flip_probability"] for k, v in
             json.loads((ROOT / "analysis" / "label_error_rates.json").read_text())["rates"].items()}
    base = conditional()

    frames = []
    for label, cfg in [("zoning label as given", replace(base, label_flip={"RM": 0.0, "NR": 0.0})),
                       ("with measured label error", base)]:
        model = StockModel(buildings, q, cfg)
        _, per_b = run_monte_carlo(model, args.n_iter, args.seed)
        df = totals_by_class(model, per_b, model.assigned).reset_index().rename(columns={"index": "material"})
        frames.append(df.melt(id_vars="material", var_name="zoning_class", value_name="stock_Mkg").assign(scenario=label))
        del per_b
    out = pd.concat(frames, ignore_index=True)
    outdir = ROOT / "analysis" / "results" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "E12_class_breakdown.csv", index=False)

    print(out.pivot_table(index=["material", "zoning_class"], columns="scenario",
                          values="stock_Mkg").round(1).to_string())
    print(f"wrote {outdir/'E12_class_breakdown.csv'}")

    try:
        import figstyle
    except ImportError:
        return
    fig, axes = figstyle.figure(1, 2, width=figstyle.DOUBLE, panel_h=3.0)
    for ax, mat in zip(axes, ["Concrete", "Brick"]):
        sub = out[out.material == mat]
        x = np.arange(len(CLASSES))
        for i, scen in enumerate(sub.scenario.unique()):
            vals = [sub[(sub.scenario == scen) & (sub.zoning_class == c)].stock_Mkg.iloc[0] for c in CLASSES]
            ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, label=scen,
                   color=figstyle.PALETTE[i], edgecolor="black")
        ax.set_xticks(x, CLASSES, rotation=20, ha="right")
        ax.set_ylabel("Study-area stock (million kg)")
        ax.set_title(f"({'ab'[list(axes).index(ax)]}) {mat}", loc="left")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
    figstyle.save(fig, "F6_stock_by_class")
    print(f"wrote {ROOT/'analysis'/'figures'/'F6_stock_by_class.png'}")


if __name__ == "__main__":
    main()
