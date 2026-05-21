import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from Monte_Carlo_Main import GeometricUncertaintyMonteCarlo
import warnings

warnings.filterwarnings('ignore')

class SampleSizeUncertaintyAnalysis:
    """
    Analyze how coefficient of variation decreases with increasing sample size.
    Uses the same MC framework but runs simulations on progressively larger building samples.
    """
    def __init__(self, data_path: str, rasmi_intensity_path: str, 
                 n_iterations: int = 3000, random_state: int = 42,
                 height_uncertainty: float = 0.15, area_uncertainty: float = 0.15,
                 enable_typology_uncertainty: bool = True, temperature: float = 1.5):
        
        self.data_path = data_path
        self.rasmi_intensity_path = rasmi_intensity_path
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.height_uncertainty = height_uncertainty
        self.area_uncertainty = area_uncertainty
        self.enable_typology_uncertainty = enable_typology_uncertainty
        self.temperature = temperature
        
        self.data = pd.read_csv(data_path)
        self.total_buildings = len(self.data)
        self.materials = ['Concrete', 'Brick', 'Wood', 'Steel', 'Glass', 'Plastics', 'Aluminium', 'Copper']
    
    def generate_sample_sizes(self, min_size: int = 50, max_size: int = None, 
                            n_samples: int = 15) -> list:
        """Generate logarithmically spaced sample sizes."""
        if max_size is None:
            max_size = self.total_buildings
        
        max_size = min(max_size, self.total_buildings)
        
        sample_sizes = np.logspace(
            np.log10(min_size), 
            np.log10(max_size), 
            num=n_samples
        )
        sample_sizes = np.unique(sample_sizes.astype(int))
        sample_sizes = sorted([s for s in sample_sizes if s <= self.total_buildings])
        
        return sample_sizes
    
    def run_sample_simulation(self, sample_size: int, sample_seed: int) -> dict:
        """Run MC simulation on a random sample of buildings."""
        np.random.seed(sample_seed)
        sample_indices = np.random.choice(self.total_buildings, size=sample_size, replace=False)
        
        sample_data = self.data.iloc[sample_indices].copy()
        sample_data = sample_data.reset_index(drop=True)
        
        temp_csv = f"temp_sample_{sample_size}_{sample_seed}.csv"
        sample_data.to_csv(temp_csv, index=False)
        
        try:
            mc = GeometricUncertaintyMonteCarlo(
                data_path=temp_csv,
                rasmi_intensity_path=self.rasmi_intensity_path,
                n_iterations=self.n_iterations,
                random_state=self.random_state,
                calibrate_probs=self.enable_typology_uncertainty,
                temperature=self.temperature,
                height_uncertainty=self.height_uncertainty,
                area_uncertainty=self.area_uncertainty,
                fix_data_issues=True
            )
            
            mc.run_simulation(save_dir=f"temp_results_{sample_size}_{sample_seed}")
            iteration_totals = mc.results.groupby('iteration')[self.materials].sum()
            total_means = iteration_totals.mean()
            total_stds = iteration_totals.std()
            total_cvs = (total_stds / total_means * 100).to_dict()
            
            Path(temp_csv).unlink(missing_ok=True)
            import shutil
            shutil.rmtree(f"temp_results_{sample_size}_{sample_seed}", ignore_errors=True)
            
            return {
                'sample_size': sample_size,
                'cvs': total_cvs,
                'means': total_means.to_dict(),
                'stds': total_stds.to_dict()
            }
        
        except Exception:
            Path(temp_csv).unlink(missing_ok=True)
            return None
    
    def run_analysis(self, sample_sizes: list = None, n_replicates: int = 3, 
                save_dir: str = "sample_size_analysis") -> pd.DataFrame:
        if sample_sizes is None:
            sample_sizes = self.generate_sample_sizes()
        
        all_results = []
        total_runs = len(sample_sizes) * n_replicates
        run_counter = 0

        for sample_size in sample_sizes:
            for rep in range(n_replicates):
                run_counter += 1
                print(f"[Progress] Running sample size {sample_size}, replicate {rep+1}/{n_replicates} "
                    f"({run_counter}/{total_runs} total)")

                result = self.run_sample_simulation(
                    sample_size=sample_size,
                    sample_seed=self.random_state + rep
                )
                
                if result:
                    result['replicate'] = rep
                    all_results.append(result)

    
        records = []
        for result in all_results:
            record = {
                'sample_size': result['sample_size'],
                'replicate': result['replicate']
            }
            for material in self.materials:
                record[f'{material}_cv'] = result['cvs'].get(material, np.nan)
                record[f'{material}_mean'] = result['means'].get(material, np.nan)
                record[f'{material}_std'] = result['stds'].get(material, np.nan)
            records.append(record)
        
        results_df = pd.DataFrame(records)
        
        summary_records = []
        for size in results_df['sample_size'].unique():
            size_data = results_df[results_df['sample_size'] == size]
            summary = {'sample_size': size}
            
            for material in self.materials:
                cv_col = f'{material}_cv'
                if cv_col in size_data.columns:
                    summary[f'{material}_cv_mean'] = size_data[cv_col].mean()
                    summary[f'{material}_cv_std'] = size_data[cv_col].std()
                    summary[f'{material}_cv_min'] = size_data[cv_col].min()
                    summary[f'{material}_cv_max'] = size_data[cv_col].max()
            
            summary_records.append(summary)
        
        summary_df = pd.DataFrame(summary_records).sort_values('sample_size')
        
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        results_df.to_csv(save_path / "all_replicates.csv", index=False)
        summary_df.to_csv(save_path / "summary_statistics.csv", index=False)
        
        return summary_df, results_df
    
    def plot_cv_vs_sample_size(self, summary_df: pd.DataFrame, 
                               save_dir: str = "sample_size_analysis"):
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for material in self.materials:
            cv_col = f'{material}_cv_mean'
            if cv_col in summary_df.columns:
                ax.plot(summary_df['sample_size'], summary_df[cv_col], 
                       marker='o', label=material, linewidth=2)
        ax.set_xlabel('Sample Size (number of buildings)', fontsize=12)
        ax.set_ylabel('Coefficient of Variation (%)', fontsize=12)
        ax.set_title('Uncertainty Reduction with Increasing Sample Size', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        plt.tight_layout()
        plt.savefig(save_path / "cv_vs_sample_size_all.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        for idx, material in enumerate(self.materials):
            ax = axes[idx]
            cv_mean_col = f'{material}_cv_mean'
            cv_std_col = f'{material}_cv_std'
            if cv_mean_col in summary_df.columns:
                x = summary_df['sample_size']
                y = summary_df[cv_mean_col]
                ax.plot(x, y, marker='o', color='steelblue', linewidth=2)
                if cv_std_col in summary_df.columns:
                    yerr = summary_df[cv_std_col]
                    ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color='steelblue')
                ax.set_xlabel('Sample Size', fontsize=10)
                ax.set_ylabel('CV (%)', fontsize=10)
                ax.set_title(material, fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_xscale('log')
        plt.suptitle('Material-Specific Uncertainty vs Sample Size', fontsize=14, y=1.00)
        plt.tight_layout()
        plt.savefig(save_path / "cv_vs_sample_size_individual.png", dpi=300, bbox_inches='tight')
        plt.close()
        
 
    def calculate_cv_reduction(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        reductions = []
        smallest = summary_df.iloc[0]
        largest = summary_df.iloc[-1]
        for material in self.materials:
            cv_col = f'{material}_cv_mean'
            if cv_col in summary_df.columns:
                cv_small = smallest[cv_col]
                cv_large = largest[cv_col]
                reduction_pct = (cv_small - cv_large) / cv_small * 100
                reductions.append({
                    'material': material,
                    'cv_small_sample': cv_small,
                    'cv_large_sample': cv_large,
                    'reduction_pct': reduction_pct,
                    'reduction_ratio': cv_small / cv_large
                })
        return pd.DataFrame(reductions)

def main():
    BASE_DIR = Path(__file__).resolve().parent
    data_path = BASE_DIR / "data" / "MC_dataset.csv"
    rasmi_intensity_path = BASE_DIR / "data" / "RASMI_intensity.csv"
    
    analyzer = SampleSizeUncertaintyAnalysis(
        data_path=str(data_path),
        rasmi_intensity_path=str(rasmi_intensity_path),
        n_iterations=3000,
        height_uncertainty=0.15,
        area_uncertainty=0.15,
        enable_typology_uncertainty=True,
        temperature=1.5
    )
    
    summary_df, results_df = analyzer.run_analysis(
        sample_sizes=[1, 2, 4, 8, 10],
        n_replicates=4,
        save_dir="sample_size_analysis"
    )
    
    analyzer.plot_cv_vs_sample_size(summary_df, save_dir="sample_size_analysis")
    reduction_df = analyzer.calculate_cv_reduction(summary_df)
    reduction_df.to_csv("sample_size_analysis/cv_reduction_summary.csv", index=False)

if __name__ == "__main__":
    main()
