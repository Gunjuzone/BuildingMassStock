"""
MI Sensitivity Analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


def _get_mi_by_percentile(mc_instance, building_idx, typology, percentile):
    """Mirrors _get_material_intensities_p50 but reads any RASMI percentile."""
    height = mc_instance.data.iloc[building_idx]['Height']
    rasmi_category = mc_instance._map_to_rasmi_category(typology, height, building_idx)
    height_category = '<=9' if height <= 9 else '>9'

    rasmi_row = mc_instance.rasmi_intensity[
        (mc_instance.rasmi_intensity['RASMI_Category'] == rasmi_category) &
        (mc_instance.rasmi_intensity['Height_Category'] == height_category)
    ]
    if rasmi_row.empty:
        rasmi_row = mc_instance.rasmi_intensity[
            (mc_instance.rasmi_intensity['RASMI_Category'] == 'RM_C') &
            (mc_instance.rasmi_intensity['Height_Category'] == '<=9')
        ]

    intensities = {}
    for material in mc_instance.materials:
        col = f'{material}_p_{percentile}'
        try:
            intensities[material] = max(0.0, float(rasmi_row[col].iloc[0]))
        except (KeyError, IndexError, ValueError):
            try:
                intensities[material] = max(0.0, float(rasmi_row[f'{material}_p_50'].iloc[0]))
            except Exception:
                intensities[material] = 0.0
    return intensities


def _run_scenario(mc_instance, percentile, n_iterations):
    """
    Run one MI scenario. Geometry + typology draws are reproduced from the
    same random seed used in the original run.

    """
    rng = np.random.RandomState(mc_instance.random_state)
    materials   = mc_instance.materials
    n_buildings = len(mc_instance.data)

    # shape: (n_iterations, n_buildings) per material
    building_masses = {mat: np.zeros((n_iterations, n_buildings)) for mat in materials}
    iteration_totals = []

    for iteration in range(n_iterations):
        if (iteration + 1) % 500 == 0:
            print(f"    MI=p{percentile} — iteration {iteration + 1}/{n_iterations}")

        iter_sums = {mat: 0.0 for mat in materials}

        for b_idx in range(n_buildings):
            bd = mc_instance.data.iloc[b_idx]

            # Typology sampling — same rng state as original run
            t_probs    = mc_instance._get_typology_probabilities(b_idx)
            typologies = list(t_probs.keys())
            probs      = [max(0.0, v) for v in t_probs.values()]
            total_p    = sum(probs)
            probs      = [p / total_p for p in probs] if total_p > 0 else [1.0 / len(typologies)] * len(typologies)
            total_p    = sum(probs)
            probs      = [p / total_p for p in probs]
            sampled_typology = rng.choice(typologies, p=probs)

            # Geometric perturbation — same rng state as original run
            h0 = bd['Height']
            a0 = bd['Area']
            h_mult   = rng.lognormal(mean=0, sigma=mc_instance.height_uncertainty)
            p_height = max(1.0, min(h0 * h_mult, h0 * 3))
            a_mult   = max(0.1, rng.normal(loc=1.0, scale=mc_instance.area_uncertainty))
            p_area   = max(a0 * 0.1, min(a0 * a_mult, a0 * 3))
            n_stories  = max(1, round(p_height / 3.0))
            floor_area = max(0.0, p_area * n_stories)

            # MI at requested percentile
            mi = _get_mi_by_percentile(mc_instance, b_idx, sampled_typology, percentile)

            for mat in materials:
                mass = floor_area * mi.get(mat, 0.0)
                building_masses[mat][iteration, b_idx] = mass
                iter_sums[mat] += mass

        row = {'iteration': iteration}
        row.update(iter_sums)
        iteration_totals.append(row)

    district_df = pd.DataFrame(iteration_totals)

    # Per-building stats
    building_rows = []
    for b_idx in range(n_buildings):
        bd  = mc_instance.data.iloc[b_idx]
        row = {
            'building_id':  bd['ID'],
            'building_idx': b_idx,
            'typology':     bd['Typology'],
            'height':       bd['Height'],
            'area':         bd['Area'],
        }
        for mat in materials:
            vals   = building_masses[mat][:, b_idx]
            mean_v = vals.mean()
            std_v  = vals.std()
            cv_v   = 100 * std_v / mean_v if mean_v > 0 else 0.0
            row[f'{mat}_mean_kg'] = round(mean_v, 4)
            row[f'{mat}_cv_pct']  = round(cv_v,   4)
        building_rows.append(row)

    building_df = pd.DataFrame(building_rows)
    return district_df, building_df


def run_mi_sensitivity_analysis(mc_instance, save_dir="MC_Results", n_iterations=None):
    """
    Run MI sensitivity at district AND building scale.

    """
    if n_iterations is None:
        n_iterations = mc_instance.n_iterations

    materials = mc_instance.materials
    scenarios = {'Low (p25)': '25', 'Base (p50)': '50', 'High (p75)': '75'}

    print("MI SENSITIVITY ANALYSIS — district + building scale")

    dist_results  = {}
    build_results = {}

    for label, pct in scenarios.items():
        print(f"\n  Scenario: {label}")
        dist_df, build_df = _run_scenario(mc_instance, pct, n_iterations)
        dist_results[label]  = dist_df
        build_results[label] = build_df

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    # District summary
    base_means = {}
    dist_rows  = []

    for label, df in dist_results.items():
        for mat in materials:
            it_sums = df[mat]
            mean_kg = it_sums.mean()
            std_kg  = it_sums.std()
            cv_pct  = 100 * std_kg / mean_kg if mean_kg > 0 else 0.0
            p5_kg   = it_sums.quantile(0.05)
            p95_kg  = it_sums.quantile(0.95)

            dist_rows.append({
                'MI_Scenario':       label,
                'Material':          mat,
                'Mean_million_kg':   round(mean_kg        / 1e6, 3),
                'Std_million_kg':    round(std_kg         / 1e6, 3),
                'CV_pct':            round(cv_pct,          2),
                'P5_million_kg':     round(p5_kg          / 1e6, 3),
                'P95_million_kg':    round(p95_kg         / 1e6, 3),
                'Range_P5_P95_Mkkg': round((p95_kg-p5_kg) / 1e6, 3),
            })

            if label == 'Base (p50)':
                base_means[mat] = mean_kg

    district_summary = pd.DataFrame(dist_rows)

    def pct_shift(row):
        base = base_means.get(row['Material'])
        if not base:
            return np.nan
        return round((row['Mean_million_kg'] * 1e6 - base) / base * 100, 2)

    district_summary['Shift_vs_Base_pct'] = district_summary.apply(pct_shift, axis=1)
    district_summary.to_csv(save_path / "mi_sensitivity_district.csv", index=False)

    pivot = district_summary.pivot_table(
        index='Material',
        columns='MI_Scenario',
        values=['Mean_million_kg', 'CV_pct', 'Range_P5_P95_Mkkg', 'Shift_vs_Base_pct']
    )
    pivot.to_csv(save_path / "mi_sensitivity_pivot.csv")

    # Building summary
    build_summary_rows = []

    for label, bdf in build_results.items():
        safe_label = label.replace(' ', '_').replace('(', '').replace(')', '')
        bdf.to_csv(save_path / f"mi_sensitivity_building_{safe_label}.csv", index=False)

        row = {'MI_Scenario': label}
        for mat in materials:
            cv_col = f'{mat}_cv_pct'
            if cv_col in bdf.columns:
                cv_vals = bdf[cv_col].replace([np.inf, -np.inf], np.nan).dropna()
                row[f'{mat}_mean_CV']   = round(cv_vals.mean(),         2)
                row[f'{mat}_median_CV'] = round(cv_vals.median(),       2)
                row[f'{mat}_p25_CV']    = round(cv_vals.quantile(0.25), 2)
                row[f'{mat}_p75_CV']    = round(cv_vals.quantile(0.75), 2)
        build_summary_rows.append(row)

    building_summary = pd.DataFrame(build_summary_rows)
    building_summary.to_csv(save_path / "mi_sensitivity_building_summary.csv", index=False)

    # Console output
    print("\n" + "=" * 95)
    print("DISTRICT-LEVEL RESULTS (million kg)")
    print("=" * 95)
    print(f"{'Material':<12} {'Low Mean':>10} {'Base Mean':>10} {'High Mean':>10} "
          f"{'Low CV%':>8} {'Base CV%':>8} {'High CV%':>8} "
          f"{'Low D%':>8} {'High D%':>8}")
    print("-" * 95)

    for mat in materials:
        low  = district_summary[(district_summary['Material'] == mat) & (district_summary['MI_Scenario'] == 'Low (p25)')]
        base = district_summary[(district_summary['Material'] == mat) & (district_summary['MI_Scenario'] == 'Base (p50)')]
        high = district_summary[(district_summary['Material'] == mat) & (district_summary['MI_Scenario'] == 'High (p75)')]
        if low.empty or base.empty or high.empty:
            continue
        print(
            f"{mat:<12}"
            f"{low['Mean_million_kg'].values[0]:>10.3f}"
            f"{base['Mean_million_kg'].values[0]:>10.3f}"
            f"{high['Mean_million_kg'].values[0]:>10.3f}"
            f"{low['CV_pct'].values[0]:>8.2f}"
            f"{base['CV_pct'].values[0]:>8.2f}"
            f"{high['CV_pct'].values[0]:>8.2f}"
            f"{low['Shift_vs_Base_pct'].values[0]:>8.2f}"
            f"{high['Shift_vs_Base_pct'].values[0]:>8.2f}"
        )


    print("BUILDING-LEVEL CV% — mean across all buildings per scenario")
    print(f"{'Material':<12} {'Low mean CV%':>14} {'Base mean CV%':>14} {'High mean CV%':>14} {'Max delta':>10}")


    for mat in materials:
        low_r  = building_summary[building_summary['MI_Scenario'] == 'Low (p25)']
        base_r = building_summary[building_summary['MI_Scenario'] == 'Base (p50)']
        high_r = building_summary[building_summary['MI_Scenario'] == 'High (p75)']
        if low_r.empty or base_r.empty or high_r.empty:
            continue
        col = f'{mat}_mean_CV'
        if col not in low_r.columns:
            continue
        lv    = low_r[col].values[0]
        bv    = base_r[col].values[0]
        hv    = high_r[col].values[0]
        delta = round(max(abs(hv - bv), abs(lv - bv)), 2)
        print(f"{mat:<12}{lv:>14.2f}{bv:>14.2f}{hv:>14.2f}{delta:>10.2f}")

    print(f"\nAll files saved to: {save_path}")
    return district_summary, building_summary