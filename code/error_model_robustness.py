"""Does the cadastre-measured error model survive a different matching rule?"""
import argparse

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from stock_engine import ROOT


def stats(g: pd.DataFrame, area_col: str) -> dict:
    r = np.log(g["Area"] * g["floors_h3"] / g[area_col])
    q25, q75 = np.percentile(r, [25, 75])
    xy = g[["X_coord", "Y_coord"]].to_numpy()
    cells = np.floor((xy - xy.min(0)) / 500).astype(int)
    blk = pd.Series(np.unique(cells, axis=0, return_inverse=True)[1].reshape(-1), index=g.index)
    ni = r.groupby(blk).size().to_numpy(float)
    gb = r.groupby(blk)
    N, k = ni.sum(), len(ni)
    if k > 1:
        msb = (ni * (gb.mean().to_numpy() - r.mean()) ** 2).sum() / (k - 1)
        msw = (gb.var(ddof=1).fillna(0).to_numpy() * (ni - 1)).sum() / max(N - k, 1)
        n0 = (N - (ni ** 2).sum() / N) / (k - 1)
        s2b = max(0.0, (msb - msw) / n0)
        between = s2b / (s2b + msw) if (s2b + msw) > 0 else np.nan
    else:
        between = np.nan
    D = squareform(pdist(xy))
    z = ((r - r.mean()) / r.std()).to_numpy()
    near = (D > 0) & (D <= 100)
    np.fill_diagonal(near, False)
    return {"n": len(g), "mean_log_ratio": r.mean(), "sd_log_ratio": r.std(ddof=1),
            "robust_sd": (q75 - q25) / 1.349,
            "corr_within_100m": np.outer(z, z)[near].mean() if near.any() else np.nan,
            "share_variance_between_500m_blocks": between}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="corrected_v1")
    args = ap.parse_args()
    m = pd.read_csv(ROOT / "data" / "validation" / "cadastre" / "study_buildings_cadastre_match.csv")
    m["floors_h3"] = np.maximum(1, np.round(m["Height"] / 3))
    rows = []

    for iou in np.round(np.arange(0.0, 0.85, 0.05), 2):
        for area_col, area_name in [("cad_parts_gfa_above_m2", "parts above ground"),
                                    ("cad_official_gfa_m2", "official GFA")]:
            g = m[(m.iou > iou) & (m.match_dist_m <= 5.0)
                  & (m.cad_max_floors_above > 0) & (m[area_col] > 0)]
            if len(g) >= 30:
                rows.append({"swept": "overlap threshold", "min_iou": iou, "max_dist_m": 5.0,
                             "area_definition": area_name, **stats(g, area_col)})

    for dist in [2.5, 5.0, 7.5, 10.0, 15.0]:
        for area_col, area_name in [("cad_parts_gfa_above_m2", "parts above ground"),
                                    ("cad_official_gfa_m2", "official GFA")]:
            g = m[(m.match_dist_m <= dist) & (m.iou > 0.3)
                  & (m.cad_max_floors_above > 0) & (m[area_col] > 0)]
            if len(g) >= 30:
                rows.append({"swept": "position tolerance", "min_iou": 0.3, "max_dist_m": dist,
                             "area_definition": area_name, **stats(g, area_col)})

    out = pd.DataFrame(rows)
    outdir = ROOT / "analysis" / "results" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "E15_error_model_robustness.csv", index=False)
    show = out[out.area_definition == "parts above ground"]
    print(show[["swept", "min_iou", "max_dist_m", "n", "mean_log_ratio", "sd_log_ratio",
                "share_variance_between_500m_blocks"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
