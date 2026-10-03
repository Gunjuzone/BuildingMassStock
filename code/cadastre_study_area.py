"""Clip the parsed Barcelona cadastre to the study districts and summarise it."""
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import wkt
from shapely.prepared import prep
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
CADASTRE = ROOT / "data" / "validation" / "cadastre" / "cadastre_buildings_barcelona.csv"
DISTRICTS = ROOT / "data" / "validation" / "boundaries" / "BarcelonaCiutat_Districtes.csv"
OUT_CSV = ROOT / "data" / "validation" / "cadastre" / "cadastre_study_districts.csv"
OUT_TXT = ROOT / "analysis" / "cadastre_study_area_summary.txt"

DISTRICT_CODES = [5, 6, 7]

PERIOD_BINS = [0, 1940, 1960, 1979, 2006, 9999]
PERIOD_LABELS = ["<=1940", "1941-1960", "1961-1979 (pre NBE-CT-79)", "1980-2006 (pre CTE)", ">=2007 (CTE)"]


def main():
    districts = pd.read_csv(DISTRICTS)
    districts = districts[districts["Codi_Districte"].isin(DISTRICT_CODES)]
    polys = {r.nom_districte: prep(wkt.loads(r.geometria_etrs89)) for r in districts.itertuples()}

    cad = pd.read_csv(CADASTRE)
    cad = cad.dropna(subset=["x_25831", "y_25831"])

    def district_of(x, y):
        pt = Point(x, y)
        return next((name for name, poly in polys.items() if poly.contains(pt)), None)

    cad["district"] = [district_of(x, y) for x, y in zip(cad["x_25831"], cad["y_25831"])]
    sub = cad[cad["district"].notna()].copy()
    sub["period"] = pd.cut(sub["year_built"], PERIOD_BINS, labels=PERIOD_LABELS)
    sub.to_csv(OUT_CSV, index=False)

    lines = [
        f"Cadastral buildings in Barcelona: {len(cad):,}",
        f"Cadastral buildings in study districts (centroid in district): {len(sub):,}",
        sub["district"].value_counts().to_string(),
        "",
        "Construction year (cadastre 'beginning'):",
        sub["year_built"].describe().round(1).to_string(),
        "",
        "Construction period:",
        sub["period"].value_counts(sort=False).to_string(),
        "",
        "Current use:",
        sub["current_use"].value_counts(dropna=False).to_string(),
        "",
        "Max floors above ground:",
        sub["max_floors_above"].describe().round(2).to_string(),
        "",
        "Share of buildings with max floors above ground <= 3 (~9 m at 3 m/floor): "
        f"{np.mean(sub['max_floors_above'] <= 3):.1%}",
        "",
        "Official gross floor area vs footprint x area-weighted floors (ratio, median): "
        f"{(sub['official_gfa_m2'] / sub['parts_gfa_above_m2']).replace([np.inf, -np.inf], np.nan).median():.2f}",
    ]
    OUT_TXT.parent.mkdir(exist_ok=True)
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
