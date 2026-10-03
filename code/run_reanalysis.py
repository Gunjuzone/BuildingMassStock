"""Scenario runs, one experiment per table of results.

  E1  MI treatment           MI at its median against MI sampled within its RASMI range, shared
                             across a category at 0 / 0.5 / 1
  E2  error magnitude        height and footprint log-SD swept from 0.05 to 0.45
  E3  storey height          fixed 2.7 / 3.0 / 3.3 m and the cadastral interquartile range
  E4  structural threshold   6 / 9 / 12 m, under the height rule the threshold belongs to
  E5  correlated errors      geometry shared globally or by 500 m block, typology and label error
                             shared within a zoning unit
  E6  sample size            study-area CV against the number of buildings, independent against
                             correlated errors
  E7  grouped Sobol          first-, total- and second-order indices, study-area and building level
  E8  label error            zoning labels flipped at the rates measured against the cadastre,
                             independently or shared within a zoning unit
  E9  structural rule        height rule against construction-period rule, with an uncertain
                             masonry share
  E10 model form             floor counts from height/3 against cadastral floor counts, over the
                             matched buildings
  E11 class mapping          Mixed-Use mapped to the RASMI residential against non-residential
                             category

Usage:
  python code/build_buildings_table.py
  python code/run_reanalysis.py --tag TAG
Results go to analysis/results/<tag>/.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from stock_engine import (CADASTRAL_STOREY_IQR, CLASS_TO_FUNCTION, CLASS_TO_FUNCTION_MU_NR, ROOT, Config, ErrorComponents, StockModel,
                          cv_by_sample_size, load_buildings, load_rasmi_ranges, run_monte_carlo, sobol_grouped,
                          summarise)

EC = ErrorComponents
from baseline import (MEASURED_FOOTPRINT, MEASURED_HEIGHT, P_MASONRY_BY_PERIOD, P_MASONRY_UNDATED,
                      classifier_probs, conditional, label_flip_rates, submitted)

SUBMITTED = submitted()
CLASSIFIER_PROBS = classifier_probs()


def mc_table(buildings, q, configs: dict[str, Config], n_iter: int, seed: int) -> pd.DataFrame:
    rows = []
    for label, cfg in configs.items():
        model = StockModel(buildings, q, cfg)
        totals, per_b = run_monte_carlo(model, n_iter, seed)
        rows.append(summarise(totals, per_b, model.deterministic()).assign(scenario=label))
        del per_b
        print(f"  {label}: done")
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buildings", type=Path, default=ROOT / "data" / "buildings" / "study_buildings_enriched.csv")
    ap.add_argument("--rasmi", type=Path, default=ROOT / "data" / "rasmi" / "RASMI_EU15.xlsx")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-iter", type=int, default=3000)
    ap.add_argument("--n-sobol", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", nargs="*", help="subset of experiments, e.g. E1 E7")
    args = ap.parse_args()

    out = ROOT / "analysis" / "results" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    buildings = load_buildings(args.buildings)
    q = load_rasmi_ranges(args.rasmi)

    global CONDITIONAL
    CONDITIONAL = conditional()
    run = set(args.only) if args.only else {f"E{i}" for i in range(1, 12)}
    meta = {"buildings": str(args.buildings), "rasmi": str(args.rasmi), "n_iter": args.n_iter,
            "n_sobol": args.n_sobol, "seed": args.seed, "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    if "E1" in run:
        print("E1 conditional vs full")
        configs = {
            "conditional_p50": CONDITIONAL,
            "full_mi_independent": replace(CONDITIONAL, mi_mode="stochastic", mi_w_category=0.0),
            "full_mi_shared_0.5": replace(CONDITIONAL, mi_mode="stochastic", mi_w_category=0.5),
            "full_mi_shared_1.0": replace(CONDITIONAL, mi_mode="stochastic", mi_w_category=1.0),
        }
        mc_table(buildings, q, configs, args.n_iter, args.seed).to_csv(out / "E1_conditional_vs_full.csv", index=False)

    if "E2" in run:
        print("E2 error magnitude")
        configs = {f"sigma_{s:.2f}": replace(CONDITIONAL, height=EC(s), footprint=EC(s)) for s in [0.05, 0.10, 0.15, 0.25, 0.35, 0.45]}
        mc_table(buildings, q, configs, args.n_iter, args.seed).to_csv(out / "E2_error_magnitude.csv", index=False)

    if "E3" in run:
        print("E3 floor height")
        configs = {f"floor_{lo:.1f}-{hi:.1f}": replace(CONDITIONAL, floor_height_range=(lo, hi))
                   for lo, hi in [(2.7, 2.7), (3.0, 3.0), (3.3, 3.3), CADASTRAL_STOREY_IQR]}
        mc_table(buildings, q, configs, args.n_iter, args.seed).to_csv(out / "E3_floor_height.csv", index=False)

    if "E4" in run:
        print("E4 structural threshold")
        height_rule = replace(CONDITIONAL, structure_rule="height", p_masonry_by_period=None,
                              p_masonry={"RM": 0.30, "NR": 0.30})
        configs = {f"threshold_{t:g}m": replace(height_rule, structure_threshold_m=t) for t in [6.0, 9.0, 12.0]}
        mc_table(buildings, q, configs, args.n_iter, args.seed).to_csv(out / "E4_threshold.csv", index=False)

    correlated = {
        "independent": (EC(0.15), EC(0.15)),
        "global_0.25": (EC(0.15, 0.25), EC(0.15, 0.25)),
        "block_0.25": (EC(0.15, 0.0, 0.25), EC(0.15, 0.0, 0.25)),
        "global_0.25_block_0.25": (EC(0.15, 0.25, 0.25), EC(0.15, 0.25, 0.25)),
        "block_0.50": (EC(0.15, 0.0, 0.50), EC(0.15, 0.0, 0.50)),
        "global_0.50": (EC(0.15, 0.50), EC(0.15, 0.50)),
        "empirical_cadastre": (EC(0.40, 0.0, 0.11), EC(0.33, 0.0, 0.06)),
        "empirical_magnitude_independent": (EC(0.40, 0.0, 0.0), EC(0.33, 0.0, 0.0)),
    }
    if "E5" in run:
        print("E5 correlated errors")
        configs = {k: replace(CONDITIONAL, height=h, footprint=a) for k, (h, a) in correlated.items()}
        flips = label_flip_rates()
        configs["classifier_probs_shared_within_unit_0.5"] = replace(CLASSIFIER_PROBS, typology_w_unit=0.5)
        configs["classifier_probs_shared_within_unit_1.0"] = replace(CLASSIFIER_PROBS, typology_w_unit=1.0)
        configs["label_error_independent"] = replace(CONDITIONAL, label_flip=flips)
        configs["label_error_shared_within_unit_0.5"] = replace(CONDITIONAL, label_flip=flips, label_w_unit=0.5)
        mc_table(buildings, q, configs, args.n_iter, args.seed).to_csv(out / "E5_correlated_errors.csv", index=False)

    if "E6" in run:
        print("E6 sample size")
        sizes = [s for s in [10, 25, 50, 100, 250, 500, 1000] if s < len(buildings)] + [len(buildings)]
        rows = []
        for k in ["independent", "block_0.25", "global_0.25_block_0.25", "empirical_cadastre"]:
            h, a = correlated[k]
            model = StockModel(buildings, q, replace(CONDITIONAL, height=h, footprint=a))
            _, per_b = run_monte_carlo(model, args.n_iter, args.seed)
            rows.append(cv_by_sample_size(per_b, sizes, n_rep=20, rng=np.random.default_rng(args.seed)).assign(scenario=k))
            del per_b
            print(f"  {k}: done")
        pd.concat(rows, ignore_index=True).to_csv(out / "E6_sample_size.csv", index=False)

    if "E7" in run:
        print("E7 grouped Sobol")
        flips = label_flip_rates()
        configs = {
            "conditional_p50_floor_cadastral_iqr": replace(CONDITIONAL, floor_height_range=CADASTRAL_STOREY_IQR, label_flip=flips),
            "full_mi_shared_0.5_floor_cadastral_iqr": replace(CONDITIONAL, mi_mode="stochastic", mi_w_category=0.5,
                                                        floor_height_range=CADASTRAL_STOREY_IQR, label_flip=flips),
            "alt_sigma_0.35_threshold_12m": replace(CONDITIONAL, height=EC(0.35), footprint=EC(0.18),
                                                    structure_threshold_m=12.0, floor_height_range=CADASTRAL_STOREY_IQR,
                                                    label_flip=flips),
            "alt_period_structure": replace(CONDITIONAL, structure_rule="period",
                                            p_masonry_by_period=P_MASONRY_BY_PERIOD,
                                            p_masonry=P_MASONRY_UNDATED, p_masonry_sd=0.10,
                                            floor_height_range=CADASTRAL_STOREY_IQR, label_flip=flips),
        }
        for label, cfg in configs.items():
            district, building, pairs = sobol_grouped(StockModel(buildings, q, cfg), n_base=args.n_sobol,
                                                      seed=args.seed, n_boot=200, building_level=True,
                                                      second_order=True)
            district.assign(scenario=label).to_csv(out / f"E7_sobol_district_{label}.csv", index=False)
            building.assign(scenario=label).to_csv(out / f"E7_sobol_building_{label}.csv", index=False)
            pairs.assign(scenario=label).to_csv(out / f"E7_sobol_pairs_{label}.csv", index=False)
            print(f"  {label}: done")

    if "E8" in run:
        print("E8 label error")
        flips = label_flip_rates()
        configs = {
            "no_label_error": replace(CONDITIONAL, label_flip={"RM": 0.0, "NR": 0.0}),
            "label_error_measured": replace(CONDITIONAL, label_flip=flips),
            "label_error_shared_unit_0.5": replace(CONDITIONAL, label_flip=flips, label_w_unit=0.5),
            "label_error_shared_unit_1.0": replace(CONDITIONAL, label_flip=flips, label_w_unit=1.0),
            "classifier_probabilities_as_submitted": CLASSIFIER_PROBS,
            "classifier_probabilities_plus_label_error": replace(CLASSIFIER_PROBS, label_flip=flips),
        }
        table = mc_table(buildings, q, configs, args.n_iter, args.seed)
        table.attrs["flip_rates"] = flips
        table.to_csv(out / "E8_label_error.csv", index=False)
        meta["label_flip_rates"] = flips

    if "E9" in run:
        print("E9 structural rule")
        configs = {
            "height_rule_9m": replace(CONDITIONAL, structure_rule="height",
                                      p_masonry_by_period=None, p_masonry={"RM": 0.30, "NR": 0.30}),
            "period_rule": replace(CONDITIONAL, structure_rule="period", p_masonry_by_period=P_MASONRY_BY_PERIOD,
                                   p_masonry=P_MASONRY_UNDATED),
            "period_and_height_rule": replace(CONDITIONAL, structure_rule="period_height",
                                              p_masonry_by_period=P_MASONRY_BY_PERIOD,
                                              p_masonry=P_MASONRY_UNDATED),
            "period_rule_uncertain_share": replace(CONDITIONAL, structure_rule="period",
                                                   p_masonry_by_period=P_MASONRY_BY_PERIOD,
                                                   p_masonry=P_MASONRY_UNDATED, p_masonry_sd=0.10),
            "height_rule_masonry_majority": replace(CONDITIONAL, p_masonry={"RM": 0.79, "NR": 0.50}),
        }
        mc_table(buildings, q, configs, args.n_iter, args.seed).to_csv(out / "E9_structural_rule.csv", index=False)

    if "E10" in run:
        print("E10 model form: floors from height vs cadastral floors")
        matched = buildings[buildings["recorded_floors"].notna()].reset_index(drop=True)
        print(f"  matched buildings with cadastral floors: {len(matched)}")
        configs = {
            "floors_from_height_round_H_over_3": CONDITIONAL,
            "floors_from_cadastre": replace(CONDITIONAL, use_recorded_floors=True),
        }
        mc_table(matched, q, configs, args.n_iter, args.seed).assign(n_buildings=len(matched)).to_csv(
            out / "E10_model_form.csv", index=False)

    if "E11" in run:
        print("E11 class-to-RASMI mapping")
        configs = {
            "mixed_use_as_residential": replace(CONDITIONAL, class_to_function=dict(CLASS_TO_FUNCTION)),
            "mixed_use_as_non_residential": replace(CONDITIONAL, class_to_function=dict(CLASS_TO_FUNCTION_MU_NR)),
        }
        mc_table(buildings, q, configs, args.n_iter, args.seed).to_csv(out / "E11_class_mapping.csv", index=False)

    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["base_config"] = asdict(CONDITIONAL)
    meta["p_masonry_by_period"] = P_MASONRY_BY_PERIOD
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"Results in {out}")


if __name__ == "__main__":
    main()
