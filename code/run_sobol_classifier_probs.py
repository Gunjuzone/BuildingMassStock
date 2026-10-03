"""Extra Sobol scenario: typology sampled from classifier probabilities."""
import json
from dataclasses import replace

from baseline import classifier_probs, with_storey_height_range
from stock_engine import CADASTRAL_STOREY_IQR, ROOT, Config, StockModel, load_buildings, load_rasmi_ranges, sobol_grouped

import os
OUT = ROOT / "analysis" / "results" / os.environ.get("TAG", "corrected_v1")
OUT.mkdir(parents=True, exist_ok=True)
SCEN = "classifier_probs_floor_cadastral_iqr"

flips = {k: v["flip_probability"]
         for k, v in json.loads((ROOT / "analysis" / "label_error_rates.json").read_text())["rates"].items()}
cfg = with_storey_height_range(classifier_probs())
buildings = load_buildings(ROOT / "data" / "buildings" / "study_buildings_enriched.csv")
district, building, pairs = sobol_grouped(StockModel(buildings, load_rasmi_ranges(), cfg),
                                          n_base=4000, seed=42, n_boot=200,
                                          building_level=True, second_order=True)
district.assign(scenario=SCEN).to_csv(OUT / f"E7_sobol_district_{SCEN}.csv", index=False)
building.assign(scenario=SCEN).to_csv(OUT / f"E7_sobol_building_{SCEN}.csv", index=False)
pairs.assign(scenario=SCEN).to_csv(OUT / f"E7_sobol_pairs_{SCEN}.csv", index=False)
print("done:", SCEN)
