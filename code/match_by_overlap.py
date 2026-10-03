"""Match study buildings to cadastral records by footprint overlap.

Usage: python code/match_by_overlap.py
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "data" / "buildings" / "study_buildings.csv"
CAD = ROOT / "data" / "validation" / "cadastre" / "cadastre_buildings_barcelona.csv"
OUT = ROOT / "data" / "validation" / "cadastre" / "study_buildings_cadastre_match.csv"
SHP = next((ROOT / "data" / "buildings").rglob("*_building.shp"))

CAD_COLS = ["cadastral_ref", "year_built", "condition", "current_use", "n_units", "n_dwellings",
            "official_gfa_m2", "footprint_m2", "x_25831", "y_25831", "n_parts", "max_floors_above",
            "max_floors_below", "parts_gfa_above_m2", "mean_floors_above_area_weighted"]


def main():
    study = pd.read_csv(STUDY)
    maxar = gpd.read_file(SHP).to_crs(25831)
    maxar = maxar.set_index("Polygon_ID").reindex(study["Polygon_ID"])
    if maxar.geometry.isna().any():
        raise SystemExit("some study buildings have no Maxar polygon")
    print(f"study buildings: {len(study)}, all with a Maxar polygon")

    cad = pd.read_csv(CAD, dtype={"cadastral_ref": str}).dropna(subset=["wkt_25831"])
    gcad = gpd.GeoDataFrame(cad, geometry=cad["wkt_25831"].map(wkt.loads), crs=25831)
    aoi = maxar.total_bounds
    pad = 200
    gcad = gcad.cx[aoi[0] - pad:aoi[2] + pad, aoi[1] - pad:aoi[3] + pad].reset_index(drop=True)
    print(f"cadastral buildings near the study area: {len(gcad):,}")

    m = gpd.GeoDataFrame({"row": range(len(maxar))}, geometry=maxar.geometry.values, crs=25831)
    hits = gpd.sjoin(m, gcad[["geometry"]], how="left", predicate="intersects")

    best = {}
    for row, grp in hits.groupby("row"):
        geom = m.geometry.iloc[row]
        top, top_iou = None, 0.0
        for j in grp["index_right"].dropna().unique():
            c = gcad.geometry.iloc[int(j)]
            u = geom.union(c).area
            iou = geom.intersection(c).area / u if u else 0.0
            if iou > top_iou:
                top, top_iou = int(j), iou
        best[row] = (top, top_iou)

    out = study.copy()
    idx = [best.get(i, (None, 0.0))[0] for i in range(len(study))]
    out["iou"] = [best.get(i, (None, 0.0))[1] for i in range(len(study))]
    for c in CAD_COLS:
        out["cad_" + c] = [gcad[c].iloc[j] if j is not None else np.nan for j in idx]

    out["match_dist_m"] = np.hypot(out["X_coord"].to_numpy() - out["cad_x_25831"].to_numpy(),
                                   out["Y_coord"].to_numpy() - out["cad_y_25831"].to_numpy())
    out.to_csv(OUT, index=False)

    print(f"\nwrote {OUT}")
    print(f"  with any overlapping cadastral building: {sum(j is not None for j in idx)}")
    for t in (0.1, 0.3, 0.5, 0.7):
        print(f"  IoU > {t}: {(out.iou > t).sum():5d}  ({100 * (out.iou > t).mean():.1f}%)")
    sel = out[out.iou > 0.5]
    print(f"\n  at IoU > 0.5: median centroid distance {sel.match_dist_m.median():.1f} m, "
          f"90th percentile {sel.match_dist_m.quantile(0.9):.1f} m")


if __name__ == "__main__":
    main()
