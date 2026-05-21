import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

class GeometricUncertaintyMonteCarlo:
    """
    Monte Carlo simulation framework for estimating urban building material stocks
    under GEOMETRIC UNCERTAINTY and TYPOLOGY UNCERTAINTY, using p50 material intensities
    from a separate RASMI dataset.
    """
    def __init__(self, data_path: str, rasmi_intensity_path: str, n_iterations: int = 3000, 
                 random_state: int = 42, calibrate_probs: bool = True, 
                 calibration_method: str = 'temperature_scaling', temperature: float = 1.5, 
                 fix_data_issues: bool = True, height_uncertainty: float = 0.15, 
                 area_uncertainty: float = 0.15):
        self.data = pd.read_csv(data_path) if data_path else None
        self.rasmi_intensity = pd.read_csv(rasmi_intensity_path) if rasmi_intensity_path else None
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.calibrate_probs = calibrate_probs
        self.calibration_method = calibration_method
        self.temperature = temperature
        self.fix_data_issues = fix_data_issues
        self.height_uncertainty = height_uncertainty
        self.area_uncertainty = area_uncertainty
        
        self.rng = np.random.RandomState(random_state)
        self.materials = ['Concrete', 'Brick', 'Wood', 'Steel', 'Glass', 'Plastics', 'Aluminium', 'Copper']
        self.typologies = ['Residential', 'Mixed-Use', 'Institutional', 'Amenities']
        self.rasmi_category_counts = Counter()
        self.baseline_rasmi_counts = Counter()
        
        if self.data is not None and self.rasmi_intensity is not None:
            self._validate_data()
            if self.fix_data_issues:
                self.data = self._fix_data_issues(self.data)
            if self.calibrate_probs:
                self.data = self._calibrate_probabilities(self.data, self.calibration_method, self.temperature)
        
        self.results = None
        self.building_results = None
    
   
    def _validate_data(self):
        """Validate that required columns are present in the datasets."""
        required_cols = ['ID', 'Height', 'Area', 'Typology', 'X_coord', 'Y_coord']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in main dataset: {missing_cols}")
        
        required_rasmi_cols = ['RASMI_Category', 'Height_Category'] + [f'{mat}_p_50' for mat in self.materials]
        missing_rasmi_cols = [col for col in required_rasmi_cols if col not in self.rasmi_intensity.columns]
        if missing_rasmi_cols:
            raise ValueError(f"Missing required columns in RASMI dataset: {missing_rasmi_cols}")
        
        if (self.data['Height'] <= 0).any():
            raise ValueError("Height values must be positive")
        if (self.data['Area'] <= 0).any():
            raise ValueError("Area values must be positive")
    
    def _fix_data_issues(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fix basic data quality issues in the main dataset."""
        data = data.copy()
        
        if self.calibrate_probs:
            prob_cols = ['Prob_Residential', 'Prob_Mixed-Use', 'Prob_Institutional', 'Prob_Amenities']
            if all(col in data.columns for col in prob_cols):
                min_prob = 1e-6
                for col in prob_cols:
                    small_prob_mask = (data[col] > 0) & (data[col] < min_prob)
                    if small_prob_mask.any():
                        data.loc[small_prob_mask, col] = min_prob
                
                prob_matrix = data[prob_cols].values
                row_sums = prob_matrix.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums == 0, 1, row_sums)
                normalized_probs = prob_matrix / row_sums
                
                for i, col in enumerate(prob_cols):
                    data[col] = normalized_probs[:, i]
        
        return data
    
    def _calibrate_probabilities(self, data: pd.DataFrame, method: str, temperature: float) -> pd.DataFrame:
        """Calibrate overconfident probabilities using temperature scaling."""
        prob_cols = ['Prob_Residential', 'Prob_Mixed-Use', 'Prob_Institutional', 'Prob_Amenities']
        
        if not all(col in data.columns for col in prob_cols):
            return data

        prob_matrix = data[prob_cols].values
        row_sums = prob_matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        prob_matrix = prob_matrix / row_sums
        
        if method == 'temperature_scaling':
            eps = 1e-8
            prob_matrix = np.clip(prob_matrix, eps, 1 - eps)
            reference_prob = prob_matrix[:, -1:]
            other_probs = prob_matrix[:, :-1]
            reference_prob = np.where(reference_prob < eps, eps, reference_prob)
            logits = np.log(other_probs / reference_prob)
            scaled_logits = logits / temperature
            exp_logits = np.exp(scaled_logits)
            calibrated_probs = np.zeros_like(prob_matrix)
            calibrated_probs[:, :-1] = exp_logits
            calibrated_probs[:, -1] = 1.0
            calibrated_probs = calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)
            calibrated_probs = np.clip(calibrated_probs, eps, 1 - eps)
            calibrated_probs = calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)
        
        data_calibrated = data.copy()
        for i, col in enumerate(prob_cols):
            data_calibrated[col] = calibrated_probs[:, i]
        
        return data_calibrated

    def analyze_calibration_impact(self, original_data: pd.DataFrame) -> pd.DataFrame:
        """Analyze the impact of probability calibration."""
        prob_cols = ['Prob_Residential', 'Prob_Mixed-Use', 'Prob_Institutional', 'Prob_Amenities']
        
        if not all(col in self.data.columns for col in prob_cols):
            return pd.DataFrame()
        
        comparison = pd.DataFrame({
            'building_id': self.data['ID'],
            'building_idx': range(len(self.data))
        })
        
        for col in prob_cols:
            original_probs = original_data[col]
            calibrated_probs = self.data[col]
            comparison[f'{col}_original'] = original_probs
            comparison[f'{col}_calibrated'] = calibrated_probs
            comparison[f'{col}_diff'] = calibrated_probs - original_probs
        
        return comparison

    def _get_typology_probabilities(self, building_idx: int) -> Dict[str, float]:
        """Extract typology probabilities for a building."""
        probabilities = {typ: 0.0 for typ in self.typologies}
        original_typology = self.data.iloc[building_idx]['Typology']
        
        if not self.calibrate_probs:
            if original_typology in probabilities:
                probabilities[original_typology] = 1.0
            else:
                probabilities[self.typologies[0]] = 1.0
            return probabilities
        
        prob_columns = [col for col in self.data.columns if col.startswith('Prob_') or col.endswith('_prob')]
        if prob_columns and all(col in self.data.columns for col in prob_columns):
            for col in prob_columns:
                class_name = col.replace('Prob_', '').replace('_prob', '')
                if class_name in probabilities:
                    prob_value = self.data.iloc[building_idx][col]
                    probabilities[class_name] = max(0.0, float(prob_value))
            
            total_prob = sum(probabilities.values())
            if total_prob > 0:
                return probabilities
            else:
                probabilities = {typ: 0.0 for typ in self.typologies}
                if original_typology in probabilities:
                    probabilities[original_typology] = 1.0
                else:
                    probabilities[self.typologies[0]] = 1.0
                return probabilities
        else:
            if original_typology in probabilities:
                probabilities[original_typology] = 1.0
            else:
                probabilities[self.typologies[0]] = 1.0
            return probabilities

    def _sample_typology(self, probabilities: Dict[str, float]) -> str:
        """Sample a typology based on probabilities."""
        typologies = list(probabilities.keys())
        probs = list(probabilities.values())
        total_prob = sum(probs)
        if total_prob > 0:
            probs = [p / total_prob for p in probs]
        else:
            probs = [1.0 / len(typologies) for _ in typologies]
        probs = [max(0.0, p) for p in probs]
        total_prob = sum(probs)
        if total_prob > 0:
            probs = [p / total_prob for p in probs]
        else:
            probs = [1.0 / len(typologies) for _ in typologies]
        try:
            return self.rng.choice(typologies, p=probs)
        except ValueError:
            return typologies[0]

    def _map_to_rasmi_category(self, typology: str, height: float, building_idx: int) -> str:
        """Map typology and height to RASMI_Category with conditional reclassification."""
        if typology not in self.typologies:
            typology = self.typologies[0]
        if height <= 9:
            if typology == 'Residential':
                return self.rng.choice(['RM_C', 'RM_M'], p=[0.7, 0.3])
            elif typology == 'Mixed-Use':
                prob_res = self.data.iloc[building_idx]['Prob_Residential']
                if prob_res > 0.5:
                    return self.rng.choice(['RM_C', 'RM_M'], p=[0.7, 0.3])
                else:
                    return self.rng.choice(['NR_C', 'NR_M'], p=[0.90, 0.10])
            elif typology == 'Institutional':
                prob_inst = self.data.iloc[building_idx]['Prob_Institutional']
                if prob_inst > 0.5:
                    return self.rng.choice(['NR_C', 'NR_M'], p=[0.7, 0.3])
                else:
                    return self.rng.choice(['NR_C', 'NR_M'], p=[0.95, 0.05])
            elif typology == 'Amenities':
                prob_amen = self.data.iloc[building_idx]['Prob_Amenities']
                if prob_amen > 0.5:
                    return self.rng.choice(['NR_C', 'NR_M'], p=[0.7, 0.3])
                else:
                    return self.rng.choice(['NR_C', 'NR_M'], p=[0.95, 0.05])
        else:
            return 'RM_C' if typology in ['Residential', 'Mixed-Use'] else 'NR_C'

    def _get_material_intensities_p50(self, building_idx: int, typology: str) -> Dict[str, float]:
        """Get p50 material intensities based on sampled typology and height."""
        intensities = {}
        height = self.data.iloc[building_idx]['Height']
        rasmi_category = self._map_to_rasmi_category(typology, height, building_idx)
        height_category = '<=9' if height <= 9 else '>9'
        
        self.rasmi_category_counts[rasmi_category] += 1
        
        rasmi_row = self.rasmi_intensity[
            (self.rasmi_intensity['RASMI_Category'] == rasmi_category) &
            (self.rasmi_intensity['Height_Category'] == height_category)
        ]
        
        if rasmi_row.empty:
            rasmi_row = self.rasmi_intensity[
                (self.rasmi_intensity['RASMI_Category'] == 'RM_C') &
                (self.rasmi_intensity['Height_Category'] == '<=9')
            ]
        
        for material in self.materials:
            try:
                p50 = float(rasmi_row[f'{material}_p_50'].iloc[0])
                intensities[material] = max(0.0, p50)
            except Exception:
                intensities[material] = 0.0
        
        return intensities

    def _perturb_geometry(self, height: float, area: float) -> Tuple[float, float]:
        """Apply geometric uncertainty to height and area."""
        try:
            height_multiplier = self.rng.lognormal(mean=0, sigma=self.height_uncertainty)
            perturbed_height = height * height_multiplier
            area_multiplier = self.rng.normal(loc=1.0, scale=self.area_uncertainty)
            area_multiplier = max(0.1, area_multiplier)
            perturbed_area = area * area_multiplier
            perturbed_height = max(1.0, min(perturbed_height, height * 3))
            perturbed_area = max(area * 0.1, min(perturbed_area, area * 3))
            return perturbed_height, perturbed_area
        except Exception:
            return height, area

    def _calculate_floor_area(self, height: float, footprint_area: float, floor_height: float = 3.0) -> float:
        """Calculate gross floor area."""
        n_stories = max(1, round(height / floor_height))
        gross_floor_area = footprint_area * n_stories
        return max(0.0, gross_floor_area)

    def _calculate_material_mass(self, floor_area: float, material_intensities: Dict[str, float]) -> Dict[str, float]:
        """Calculate material masses."""
        material_masses = {}
        for material, intensity in material_intensities.items():
            try:
                mass = floor_area * max(0.0, intensity)
                material_masses[material] = max(0.0, mass)
            except Exception:
                material_masses[material] = 0.0
        return material_masses

    def run_simulation(self, save_dir: str = "geometric_mc_results") -> pd.DataFrame:
        """Run Monte Carlo simulation with geometric and optional typology uncertainty."""
        if self.data is None or self.rasmi_intensity is None:
            raise ValueError("Main dataset or RASMI intensity dataset not loaded.")
        
        print("RUNNING MONTE CARLO SIMULATION")
        
        n_buildings = len(self.data)
        all_results = []
        failed_calculations = 0

        for iteration in range(self.n_iterations):
            if (iteration + 1) % 100 == 0:
                print(f"Progress: {iteration + 1}/{self.n_iterations}")
            iteration_results = []

            for building_idx in range(n_buildings):
                try:
                    building_data = self.data.iloc[building_idx]
                    typology_probs = self._get_typology_probabilities(building_idx)
                    sampled_typology = self._sample_typology(typology_probs)
                    original_height = building_data['Height']
                    original_area = building_data['Area']
                    perturbed_height, perturbed_area = self._perturb_geometry(original_height, original_area)
                    floor_area = self._calculate_floor_area(perturbed_height, perturbed_area)
                    material_intensities = self._get_material_intensities_p50(building_idx, sampled_typology)
                    material_masses = self._calculate_material_mass(floor_area, material_intensities)

                    result = {
                        'iteration': iteration,
                        'building_id': building_data['ID'],
                        'building_idx': building_idx,
                        'sampled_typology': sampled_typology,
                        'original_height': original_height,
                        'perturbed_height': perturbed_height,
                        'height_ratio': perturbed_height / original_height if original_height > 0 else 1.0,
                        'original_area': original_area,
                        'perturbed_area': perturbed_area,
                        'area_ratio': perturbed_area / original_area if original_area > 0 else 1.0,
                        'floor_area': floor_area,
                        'original_floor_area': self._calculate_floor_area(original_height, original_area),
                        **material_masses
                    }
                    iteration_results.append(result)
                except Exception:
                    failed_calculations += 1
                    building_data = self.data.iloc[building_idx]
                    result = {
                        'iteration': iteration,
                        'building_id': building_data['ID'],
                        'building_idx': building_idx,
                        'sampled_typology': building_data['Typology'],
                        'original_height': building_data['Height'],
                        'perturbed_height': building_data['Height'],
                        'height_ratio': 1.0,
                        'original_area': building_data['Area'],
                        'perturbed_area': building_data['Area'],
                        'area_ratio': 1.0,
                        'floor_area': self._calculate_floor_area(building_data['Height'], building_data['Area']),
                        'original_floor_area': self._calculate_floor_area(building_data['Height'], building_data['Area']),
                        **{material: 0.0 for material in self.materials}
                    }
                    iteration_results.append(result)
            
            all_results.extend(iteration_results)

        results_df = pd.DataFrame(all_results)
        building_stats = self._calculate_building_statistics(results_df)

        self.results = results_df
        self.building_results = building_stats

        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        results_df.to_csv(save_path / "iteration_results.csv", index=False)
        building_stats.to_csv(save_path / "building_statistics.csv", index=False)

        return building_stats

    def _calculate_building_statistics(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate building-level statistics."""
        building_groups = results_df.groupby('building_idx')
        building_stats = []

        for building_idx, group in building_groups:
            try:
                building_data = self.data.iloc[building_idx]
                stats = {
                    'building_id': building_data['ID'],
                    'building_idx': building_idx,
                    'x_coord': building_data['X_coord'],
                    'y_coord': building_data['Y_coord'],
                    'original_typology': building_data['Typology'],
                    'original_height': building_data['Height'],
                    'original_area': building_data['Area'],
                }

                for param in ['height', 'area', 'floor_area']:
                    col_name = f'perturbed_{param}' if param != 'floor_area' else param
                    if col_name in group.columns:
                        values = group[col_name].dropna()
                        if len(values) > 0:
                            mean_val = values.mean()
                            std_val = values.std()
                            stats.update({
                                f'{param}_mean': mean_val,
                                f'{param}_std': std_val if not pd.isna(std_val) else 0.0,
                                f'{param}_cv': std_val / mean_val if mean_val > 0 and not pd.isna(std_val) else 0.0,
                            })
                        else:
                            stats.update({
                                f'{param}_mean': 0.0,
                                f'{param}_std': 0.0,
                                f'{param}_cv': 0.0,
                            })

                for param in ['height_ratio', 'area_ratio']:
                    if param in group.columns:
                        values = group[param].dropna()
                        if len(values) > 0:
                            mean_val = values.mean()
                            std_val = values.std()
                            stats.update({
                                f'{param}_mean': mean_val,
                                f'{param}_std': std_val if not pd.isna(std_val) else 0.0,
                                f'{param}_cv': std_val / mean_val if mean_val > 0 and not pd.isna(std_val) else 0.0,
                            })

                for material in self.materials:
                    if material in group.columns:
                        values = group[material].dropna()
                        if len(values) > 0:
                            mean_val = values.mean()
                            std_val = values.std()
                            stats.update({
                                f'{material}_mean': mean_val,
                                f'{material}_std': std_val if not pd.isna(std_val) else 0.0,
                                f'{material}_cv': std_val / mean_val if mean_val > 0 and not pd.isna(std_val) else 0.0,
                                f'{material}_p5': values.quantile(0.05),
                                f'{material}_p25': values.quantile(0.25),
                                f'{material}_p50': values.quantile(0.50),
                                f'{material}_p75': values.quantile(0.75),
                                f'{material}_p95': values.quantile(0.95),
                            })
                        else:
                            stats.update({
                                f'{material}_mean': 0.0,
                                f'{material}_std': 0.0,
                                f'{material}_cv': 0.0,
                                f'{material}_p5': 0.0,
                                f'{material}_p25': 0.0,
                                f'{material}_p50': 0.0,
                                f'{material}_p75': 0.0,
                                f'{material}_p95': 0.0,
                            })

                if 'sampled_typology' in group.columns:
                    typology_counts = group['sampled_typology'].value_counts()
                    total_samples = len(group)
                    for typology in self.typologies:
                        stats[f'typology_prob_{typology}'] = typology_counts.get(typology, 0) / total_samples if total_samples > 0 else 0.0

                building_stats.append(stats)
            except Exception:
                building_data = self.data.iloc[building_idx]
                stats = {
                    'building_id': building_data['ID'],
                    'building_idx': building_idx,
                    'x_coord': building_data.get('X_coord', 0),
                    'y_coord': building_data.get('Y_coord', 0),
                    'original_typology': building_data.get('Typology', 'Unknown'),
                    'original_height': building_data.get('Height', 0),
                    'original_area': building_data.get('Area', 0),
                }
                building_stats.append(stats)

        return pd.DataFrame(building_stats)

    def get_deterministic_baseline(self) -> pd.DataFrame:
        """Calculate deterministic baseline using p50 MI values and original geometry."""
        if self.data is None or self.rasmi_intensity is None:
            raise ValueError("Main dataset or RASMI intensity dataset not loaded.")
        
        self.baseline_rasmi_counts = Counter()
        baseline_results = []
        
        for building_idx in range(len(self.data)):
            try:
                building_data = self.data.iloc[building_idx]
                height = building_data['Height']
                area = building_data['Area']
                floor_area = self._calculate_floor_area(height, area)
                typology = building_data['Typology']
                material_intensities = self._get_material_intensities_p50(building_idx, typology)
                material_masses = self._calculate_material_mass(floor_area, material_intensities)
                
                result = {
                    'building_id': building_data['ID'],
                    'building_idx': building_idx,
                    'typology': typology,
                    'height': height,
                    'area': area,
                    'floor_area': floor_area,
                    'rasmi_category': self._map_to_rasmi_category(typology, height, building_idx),
                    **material_masses
                }
                baseline_results.append(result)
                self.baseline_rasmi_counts[result['rasmi_category']] += 1
            except Exception:
                pass
        
        return pd.DataFrame(baseline_results)

    def calculate_uncertainty_metrics(self, save_dir: str = "geometric_mc_results") -> pd.DataFrame:
        """Calculate uncertainty metrics."""
        if self.building_results is None:
            raise ValueError("Run simulation first.")
        
        deterministic = self.get_deterministic_baseline()
        comparison = self.building_results.merge(
            deterministic, on=['building_id', 'building_idx'], suffixes=('_prob', '_det'), how='left'
        )

        for material in self.materials:
            if f'{material}_mean' in comparison.columns and material in comparison.columns:
                prob_mean = comparison[f'{material}_mean']
                det_value = comparison[material]
                prob_std = comparison[f'{material}_std']
                comparison[f'{material}_rel_diff'] = np.where(
                    det_value != 0, 
                    (prob_mean - det_value) / det_value * 100,
                    0
                )
                comparison[f'{material}_uncertainty'] = np.where(
                    prob_mean != 0,
                    prob_std / prob_mean * 100,
                    0
                )
                if f'{material}_p95' in comparison.columns and f'{material}_p5' in comparison.columns:
                    comparison[f'{material}_ci_width'] = np.where(
                        prob_mean != 0,
                        (comparison[f'{material}_p95'] - comparison[f'{material}_p5']) / prob_mean * 100,
                        0
                    )

        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        deterministic.to_csv(save_path / "deterministic_baseline.csv", index=False)
        comparison.to_csv(save_path / "uncertainty_metrics.csv", index=False)
        
        return comparison

    def calculate_typology_mass_stock(self, save_dir: str = "geometric_mc_results") -> pd.DataFrame:
        """Calculate typology-level material stock with 95% CI."""
        if self.building_results is None or self.results is None:
            raise ValueError("Run simulation first.")
        
        deterministic = self.get_deterministic_baseline()
        typology_groups = self.results.groupby('sampled_typology')
        typology_stats = []

        for typology, group in typology_groups:
            stats = {'typology': typology}
            det_group = deterministic[deterministic['typology'] == typology]
            for material in self.materials:
                if material in group.columns:
                    iteration_sums = group.groupby('iteration')[material].sum()
                    stats[f'{material}_prob_mass'] = iteration_sums.mean()
                    stats[f'{material}_prob_mass_p2_5'] = iteration_sums.quantile(0.025)
                    stats[f'{material}_prob_mass_p97_5'] = iteration_sums.quantile(0.975)
                if material in det_group.columns:
                    stats[f'{material}_det_mass'] = det_group[material].sum()
            typology_stats.append(stats)
        
        typology_df = pd.DataFrame(typology_stats)
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        typology_df.to_csv(save_path / "typology_mass_stock.csv", index=False)
        
        return typology_df

    def calculate_total_mass_stock(self, save_dir: str = "geometric_mc_results") -> pd.DataFrame:
        """Calculate total material stock with 95% CI."""
        if self.results is None:
            raise ValueError("Run simulation first.")
        
        deterministic = self.get_deterministic_baseline()
        total_stats = {'category': 'total'}

        for material in self.materials:
            if material in self.results.columns:
                iteration_sums = self.results.groupby('iteration')[material].sum()
                total_stats[f'{material}_prob_mass'] = iteration_sums.mean()
                total_stats[f'{material}_prob_mass_p2_5'] = iteration_sums.quantile(0.025)
                total_stats[f'{material}_prob_mass_p97_5'] = iteration_sums.quantile(0.975)
            if material in deterministic.columns:
                total_stats[f'{material}_det_mass'] = deterministic[material].sum()
        
        total_df = pd.DataFrame([total_stats])
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        total_df.to_csv(save_path / "total_mass_stock.csv", index=False)
        
        return total_df

    def calculate_typology_uncertainty(self, save_dir: str = "geometric_mc_results") -> pd.DataFrame:
        """Calculate typology-level uncertainty (CV%)."""
        if self.building_results is None:
            raise ValueError("Run simulation first.")
        
        typology_groups = self.building_results.groupby('original_typology')
        typology_stats = []

        for typology, group in typology_groups:
            stats = {'typology': typology}
            for material in self.materials:
                if f'{material}_cv' in group.columns:
                    mean_cv = group[f'{material}_cv'].mean() * 100
                    stats[f'{material}_cv'] = mean_cv
            typology_stats.append(stats)
        
        typology_df = pd.DataFrame(typology_stats)
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        x = np.arange(len(typology_df))
        width = 0.1
        for i, material in enumerate(self.materials):
            if f'{material}_cv' in typology_df.columns:
                plt.bar(x + i*width, typology_df[f'{material}_cv'], width, label=material)
        plt.xlabel('Typology')
        plt.ylabel('Coefficient of Variation (CV%)')
        plt.title('Typology-Level Uncertainty by Material')
        plt.xticks(x + width*3.5, typology_df['typology'])
        plt.legend()
        plt.savefig(save_path / "typology_uncertainty_plot.png")
        plt.close()
        
        typology_df.to_csv(save_path / "typology_uncertainty.csv", index=False)
        
        return typology_df

    def calculate_total_uncertainty(self):
        """Calculate total-level coefficient of variation."""
        if self.results is None:
            raise ValueError("Run simulation first.")
        try:
            total_per_iteration = self.results.groupby('iteration')[self.materials].sum()
            total_mean = total_per_iteration.mean()
            total_std = total_per_iteration.std()
            total_cv = 100 * total_std / total_mean
            return total_cv
        except Exception:
            return pd.Series({material: 0.0 for material in self.materials})

def main_geometric_only(data_path: str, rasmi_intensity_path: str, n_iterations: int = 3000, 
                       temperature: float = 1.5, save_dir: str = "geometric_mc_results", 
                       height_uncertainty: float = 0.15, area_uncertainty: float = 0.15,
                       enable_typology_uncertainty: bool = True):
    """
    Execute Monte Carlo simulation with geometric and optional typology uncertainty.
    """
    mc = GeometricUncertaintyMonteCarlo(
        data_path=data_path, 
        rasmi_intensity_path=rasmi_intensity_path,
        n_iterations=n_iterations, 
        calibrate_probs=enable_typology_uncertainty, 
        calibration_method='temperature_scaling', 
        temperature=temperature,
        fix_data_issues=True,
        height_uncertainty=height_uncertainty,
        area_uncertainty=area_uncertainty
    )
    
    data = pd.read_csv(data_path)
    calibration_results = mc.analyze_calibration_impact(original_data=data.copy()) if enable_typology_uncertainty else None
    
    building_stats = mc.run_simulation(save_dir=save_dir)
    uncertainty_metrics = mc.calculate_uncertainty_metrics(save_dir=save_dir)
    total_cv = mc.calculate_total_uncertainty()
    typology_uncertainty = mc.calculate_typology_uncertainty(save_dir=save_dir)
    typology_mass = mc.calculate_typology_mass_stock(save_dir=save_dir)
    total_mass = mc.calculate_total_mass_stock(save_dir=save_dir)
    
    return mc, building_stats, uncertainty_metrics, calibration_results

def analyze_geometric_vs_full_uncertainty(geometric_results_dir: str = "geometric_mc_results",
                                        full_results_dir: str = "mc_results"):
    """Compare geometric-only vs full uncertainty simulations."""
    geom_path = Path(geometric_results_dir)
    full_path = Path(full_results_dir)
    geom_metrics = pd.read_csv(geom_path / "uncertainty_metrics.csv")
    full_metrics = pd.read_csv(full_path / "uncertainty_metrics.csv")
    
    return geom_metrics, full_metrics

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    data_path = os.path.join(DATA_DIR, "MC_dataset.csv")
    rasmi_intensity_path = os.path.join(DATA_DIR, "RASMI_intensity.csv")

    mc, building_stats, uncertainty_metrics, calibration_results = main_geometric_only(S
        data_path=data_path,
        rasmi_intensity_path=rasmi_intensity_path,
        n_iterations=3000,
        temperature=1.5,
        save_dir="MC_Results",
        height_uncertainty=0.15,
        area_uncertainty=0.15,
        enable_typology_uncertainty=True
    )
    # MI SENSITIVITY ANALYSIS
    from mi_sensitivity import run_mi_sensitivity_analysis
    district_summary, building_summary = run_mi_sensitivity_analysis(mc, save_dir="MC_Results")
