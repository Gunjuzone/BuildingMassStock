"""Extract building attributes from the Spanish Cadastre INSPIRE Buildings GML (Barcelona, 08900)."""
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from lxml import etree
from shapely.geometry import Polygon
from shapely.ops import unary_union

CAD_DIR = Path(__file__).resolve().parents[1] / "data" / "validation" / "cadastre"
BUILDING_GML = CAD_DIR / "A.ES.SDGC.BU.08900.building.gml"
PART_GML = CAD_DIR / "A.ES.SDGC.BU.08900.buildingpart.gml"
OUT_CSV = CAD_DIR / "cadastre_buildings_barcelona.csv"


def text(el, path):
    found = el.find(path)
    if found is None or found.text is None:
        return None
    return found.text.strip() or None


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def year(value):
    """Cadastre dates are ISO strings; unknown years appear as e.g."""
    if value and value[:4].isdigit():
        return int(value[:4])
    return None


def ring(pos_list_el):
    vals = [float(v) for v in pos_list_el.text.split()]
    return list(zip(vals[0::2], vals[1::2]))


def geometry(el):
    """Union of all polygon patches (exterior minus interior rings)."""
    polys = []
    for patch in el.iterfind(".//{*}PolygonPatch"):
        ext = patch.find("{*}exterior//{*}posList")
        if ext is None:
            continue
        holes = [ring(p) for p in patch.iterfind("{*}interior//{*}posList")]
        poly = Polygon(ring(ext), holes)
        if not poly.is_valid:
            poly = poly.buffer(0)
        polys.append(poly)
    if not polys:
        return None
    return unary_union(polys) if len(polys) > 1 else polys[0]


def iter_features(path, tag):
    for _, el in etree.iterparse(str(path), events=("end",), tag=f"{{*}}{tag}", huge_tree=True):
        yield el
        el.clear()
        while el.getprevious() is not None:
            del el.getparent()[0]


def parse_parts():
    parts = defaultdict(lambda: {"n_parts": 0, "max_floors": 0, "gfa_above": 0.0,
                                 "part_area": 0.0, "max_floors_below": 0})
    for i, el in enumerate(iter_features(PART_GML, "BuildingPart"), 1):
        local_id = text(el, ".//{*}inspireId//{*}localId")
        if local_id is None:
            continue
        bid = local_id.split("_part")[0]
        floors = to_float(text(el, "{*}numberOfFloorsAboveGround")) or 0.0
        below = to_float(text(el, "{*}numberOfFloorsBelowGround")) or 0.0
        geom = geometry(el)
        area = geom.area if geom is not None else 0.0
        rec = parts[bid]
        rec["n_parts"] += 1
        rec["max_floors"] = max(rec["max_floors"], floors)
        rec["max_floors_below"] = max(rec["max_floors_below"], below)
        rec["gfa_above"] += area * floors
        rec["part_area"] += area
        if i % 100000 == 0:
            print(f"  parts parsed: {i:,}", file=sys.stderr)
    return parts


def parse_buildings(parts):
    rows = []
    for i, el in enumerate(iter_features(BUILDING_GML, "Building"), 1):
        bid = text(el, ".//{*}inspireId//{*}localId")
        begin = text(el, "{*}dateOfConstruction//{*}beginning")
        geom = geometry(el)
        p = parts.get(bid, {})
        part_area = p.get("part_area", 0.0)
        rows.append({
            "cadastral_ref": bid,
            "year_built": year(begin),
            "condition": text(el, "{*}conditionOfConstruction"),
            "current_use": text(el, "{*}currentUse"),
            "n_units": to_float(text(el, "{*}numberOfBuildingUnits")),
            "n_dwellings": to_float(text(el, "{*}numberOfDwellings")),
            "official_gfa_m2": to_float(text(el, "{*}officialArea//{*}value")),
            "footprint_m2": geom.area if geom is not None else None,
            "x_25831": geom.centroid.x if geom is not None else None,
            "y_25831": geom.centroid.y if geom is not None else None,
            "wkt_25831": geom.wkt if geom is not None else None,
            "n_parts": p.get("n_parts", 0),
            "max_floors_above": p.get("max_floors"),
            "max_floors_below": p.get("max_floors_below"),
            "parts_gfa_above_m2": p.get("gfa_above"),
            "mean_floors_above_area_weighted": (p["gfa_above"] / part_area) if part_area else None,
        })
        if i % 50000 == 0:
            print(f"  buildings parsed: {i:,}", file=sys.stderr)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Parsing building parts...", file=sys.stderr)
    parts = parse_parts()
    print(f"Parsing buildings ({len(parts):,} with parts)...", file=sys.stderr)
    df = parse_buildings(parts)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df):,} buildings to {OUT_CSV}", file=sys.stderr)
