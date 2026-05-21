"""
This module implements a cross-validation pipeline for multiclass
classification using AutoGluon with temperature scaling for probability calibration.

Features:
- Stratified K-fold cross-validation
- PCA-based feature selection (per fold)
- SMOTE oversampling for class imbalance
- Temperature scaling for probability calibration
- Comprehensive visualization and metrics reporting
"""

import os
import warnings
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, 
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from autogluon.tabular import TabularPredictor
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore')

# Configuration
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({
    'font.size': 16, 
    'axes.titlesize': 17, 
    'axes.labelsize': 18,
    'xtick.labelsize': 17, 
    'ytick.labelsize': 17, 
    'legend.fontsize': 15
})

# Base directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "data", "AutoML_Sample_Dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "MC")
TARGET = "Typology"
EXCLUDE_COLUMNS = ['ID']
TIME_LIMIT = 7200  # seconds
N_SPLITS = 5
N_FEATURES = 12



class TemperatureScaling:
    """
    Temperature scaling for probability calibration in multiclass classification.
    
    This post-processing method scales model logits by a learned temperature parameter
    to improve calibration without affecting accuracy.
     
    """
    
    def __init__(self):
        self.temperature = 1.0
        self.classes_ = None
    
    def fit(self, probabilities: pd.DataFrame, labels: np.ndarray) -> 'TemperatureScaling':
    
        if isinstance(probabilities, pd.DataFrame):
            self.classes_ = sorted(probabilities.columns)
            probs = probabilities[self.classes_].values
        else:
            probs = np.array(probabilities)
            self.classes_ = sorted(np.unique(labels))
        
        label_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        label_indices = np.array([label_to_idx[label] for label in labels])
        
        epsilon = 1e-8
        probs_clipped = np.clip(probs, epsilon, 1 - epsilon)
        logits = np.log(probs_clipped)
        
        def negative_log_likelihood(temp: float) -> float:
            if temp <= 0:
                return np.inf
            scaled_logits = logits / temp
            scaled_probs = np.exp(scaled_logits) / np.sum(
                np.exp(scaled_logits), axis=1, keepdims=True
            )
            scaled_probs = np.clip(scaled_probs, epsilon, 1 - epsilon)
            return -np.mean(np.log(scaled_probs[range(len(label_indices)), label_indices]))
        
        result = minimize_scalar(
            negative_log_likelihood, 
            bounds=(0.1, 5.0), 
            method='bounded'
        )
        self.temperature = result.x
        return self
    
    def transform(self, probabilities: pd.DataFrame) -> pd.DataFrame:
        """
        Apply temperature scaling to probabilities.
              
        """
        if isinstance(probabilities, pd.DataFrame):
            probs = probabilities[self.classes_].values
        else:
            probs = np.array(probabilities)
        
        epsilon = 1e-8
        probs_clipped = np.clip(probs, epsilon, 1 - epsilon)
        logits = np.log(probs_clipped)
        
        scaled_logits = logits / self.temperature
        scaled_probs = np.exp(scaled_logits) / np.sum(
            np.exp(scaled_logits), axis=1, keepdims=True
        )
        
        if isinstance(probabilities, pd.DataFrame):
            return pd.DataFrame(
                scaled_probs, 
                columns=self.classes_, 
                index=probabilities.index
            )
        return scaled_probs


def calculate_expected_calibration_error(
    labels: np.ndarray, 
    probabilities: pd.DataFrame, 
    n_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
     
    Args:
        labels: True class labels
        probabilities: Predicted class probabilities
        n_bins: Number of bins for calibration assessment
        
    Returns:
        ece: Expected Calibration Error
    """
    if isinstance(probabilities, pd.DataFrame):
        max_probs = probabilities.max(axis=1).values
        predictions = probabilities.idxmax(axis=1).values
        correct = (predictions == labels).astype(int)
    else:
        max_probs = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        correct = (predictions == labels).astype(int)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
        proportion_in_bin = in_bin.mean()
        
        if proportion_in_bin > 0:
            accuracy_in_bin = correct[in_bin].mean()
            confidence_in_bin = max_probs[in_bin].mean()
            ece += np.abs(confidence_in_bin - accuracy_in_bin) * proportion_in_bin
    
    return ece


def load_data(filepath: str, target_col: str) -> pd.DataFrame:
 

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    df = df[df[target_col] != -1].reset_index(drop=True)
    df[target_col] = df[target_col].astype(str)
    
    if 'DISTRICTE' in df.columns:
        df['DISTRICTE'] = df['DISTRICTE'].fillna('Unknown').astype('category')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    zero_variance_cols = [col for col in numeric_cols if df[col].var() == 0]
    if zero_variance_cols:
        df = df.drop(columns=zero_variance_cols)
    
    return df


def select_features_pca(
    X_train: pd.DataFrame, 
    n_features: int = 12, 
    variance_threshold: float = 0.95,
    fold_id: int = 1,
    save_visualizations: bool = False,
    output_path: str = None
) -> Tuple[List[str], pd.Series, np.ndarray, np.ndarray]:
    """
    Select top features using PCA on training data only.
    
    Args:
        X_train: Training features
        n_features: Number of features to select
        variance_threshold: Cumulative variance threshold for PCA
        fold_id: Current fold number
        save_visualizations: Whether to save PCA plots
        output_path: Directory for saving visualizations
        
    Returns:
        selected_features: List of selected feature names
        feature_scores: Feature importance scores
        explained_variance: Variance explained by each component
        cumulative_variance: Cumulative variance explained
    """
    n_features = min(n_features, X_train.shape[1])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    pca = PCA()
    pca.fit(X_scaled)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    n_components = min(
        np.argmax(cumulative_variance >= variance_threshold) + 1, 
        X_train.shape[1]
    )
    
    if save_visualizations and fold_id == 1 and output_path:
        plot_pca_scree(
            explained_variance, 
            cumulative_variance, 
            variance_threshold,
            fold_id, 
            output_path
        )
    
    pca_final = PCA(n_components=n_components)
    pca_final.fit(X_scaled)
    
    loadings = pd.DataFrame(
        pca_final.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=X_train.columns
    )
    
    if save_visualizations and fold_id == 1 and output_path:
        plot_pca_loadings(loadings, fold_id, output_path)
    
    feature_scores = loadings.abs().sum(axis=1).sort_values(ascending=False)
    selected_features = feature_scores.head(n_features).index.tolist()
    
    return selected_features, feature_scores, explained_variance, cumulative_variance


def plot_pca_scree(
    explained_var: np.ndarray,
    cumulative_var: np.ndarray,
    threshold: float,
    fold_id: int,
    output_path: str
) -> None:
    """Generate and save PCA scree plot."""
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(explained_var) + 1), 
        explained_var, 
        marker='o', 
        label='Explained Variance Ratio'
    )
    plt.plot(
        range(1, len(cumulative_var) + 1), 
        cumulative_var, 
        marker='o', 
        label='Cumulative Variance Ratio'
    )
    plt.axhline(
        y=threshold, 
        color='r', 
        linestyle='--', 
        label=f'{int(threshold*100)}% Variance Threshold'
    )
    plt.title(f'PCA Scree Plot - Fold {fold_id}')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Ratio')
    plt.legend()
    plt.grid(True)
    
    filepath = os.path.join(output_path, f'pca_scree_plot_fold_{fold_id}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pca_loadings(
    loadings: pd.DataFrame,
    fold_id: int,
    output_path: str
) -> None:
    """Generate and save PCA loadings heatmap."""
    plt.figure(figsize=(12, max(8, len(loadings) * 0.3)))
    sns.heatmap(
        loadings, 
        cmap='coolwarm', 
        center=0, 
        annot=True, 
        fmt='.2f',
        annot_kws={"size": 18},
        cbar_kws={'label': 'Loading Value'}
    )
    plt.title(f'PCA Loadings Heatmap - Fold {fold_id}')
    plt.xlabel('Principal Components')
    plt.ylabel('Features')
    
    filepath = os.path.join(output_path, f'pca_loadings_heatmap_fold_{fold_id}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def aggregate_feature_analysis(
    feature_importances: List[pd.Series],
    selected_features: List[List[str]],
    explained_variances: List[np.ndarray],
    cumulative_variances: List[np.ndarray],
    n_splits: int,
    output_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create aggregated feature analysis visualizations across folds.
    
    Args:
        feature_importances: Feature importance scores from each fold
        selected_features: Selected features from each fold
        explained_variances: PCA explained variance from each fold
        cumulative_variances: PCA cumulative variance from each fold
        n_splits: Number of CV splits
        output_path: Directory for saving visualizations
        
    Returns:
        importance_summary: Aggregated feature importance DataFrame
        consistency_summary: Feature selection consistency DataFrame
    """
    all_features = set()
    for features in selected_features:
        all_features.update(features)
    
    avg_importance = {}
    for feature in all_features:
        importances = [
            fold_imp.get(feature, 0.0) 
            for fold_imp in feature_importances
        ]
        avg_importance[feature] = np.mean(importances)
    
    importance_summary = pd.DataFrame(
        list(avg_importance.items()),
        columns=['Feature', 'Average_Importance']
    ).sort_values('Average_Importance', ascending=False)
    
    plot_aggregated_importance(importance_summary, output_path)
    
    selection_counts = {}
    for features in selected_features:
        for feature in features:
            selection_counts[feature] = selection_counts.get(feature, 0) + 1
    
    consistency_summary = pd.DataFrame(
        list(selection_counts.items()),
        columns=['Feature', 'Selection_Count']
    ).sort_values('Selection_Count', ascending=False)
    
    plot_selection_consistency(consistency_summary, n_splits, output_path)
    
    if explained_variances and cumulative_variances:
        plot_average_variance(
            explained_variances, 
            cumulative_variances, 
            output_path
        )
    
    return importance_summary, consistency_summary


def plot_aggregated_importance(
    importance_df: pd.DataFrame,
    output_path: str,
    top_n: int = 20
) -> None:
    """Generate aggregated feature importance plot."""
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(top_n)
    plt.barh(range(len(top_features)), top_features['Average_Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Average Feature Importance Across Folds')
    plt.title('Top 20 Features by Average PCA Contribution')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    filepath = os.path.join(output_path, 'aggregated_feature_importance.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_selection_consistency(
    consistency_df: pd.DataFrame,
    n_splits: int,
    output_path: str,
    top_n: int = 20
) -> None:
    """Generate feature selection consistency plot."""
    plt.figure(figsize=(12, 8))
    top_features = consistency_df.head(top_n)
    colors = [
        'red' if count == n_splits else 'blue' 
        for count in top_features['Selection_Count']
    ]
    plt.barh(range(len(top_features)), top_features['Selection_Count'], color=colors)
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Number of Folds Feature Was Selected')
    plt.title(
        f'Feature Selection Consistency Across {n_splits} Folds\n'
        f'(Red = Selected in all folds)'
    )
    plt.axvline(x=n_splits, color='red', linestyle='--', alpha=0.7, label='All folds')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    
    filepath = os.path.join(output_path, 'feature_selection_consistency.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_average_variance(
    explained_variances: List[np.ndarray],
    cumulative_variances: List[np.ndarray],
    output_path: str
) -> None:
    """Generate average PCA variance explained plot."""
    min_length = min(len(var) for var in explained_variances)
    avg_explained = np.mean([var[:min_length] for var in explained_variances], axis=0)
    avg_cumulative = np.mean([cum[:min_length] for cum in cumulative_variances], axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(avg_explained) + 1), 
        avg_explained, 
        marker='o', 
        label='Avg Explained Variance Ratio'
    )
    plt.plot(
        range(1, len(avg_cumulative) + 1), 
        avg_cumulative, 
        marker='o', 
        label='Avg Cumulative Variance Ratio'
    )
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance Threshold')
    plt.title('Average PCA Variance Explained Across All Folds')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Ratio')
    plt.legend()
    plt.grid(True)
    
    filepath = os.path.join(output_path, 'pca_average_scree_plot.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def train_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fold_id: int,
    n_features: int,
    time_budget: int,
    output_path: str,
    target_col: str
) -> Tuple[np.ndarray, pd.DataFrame, float, List[str], pd.Series, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Train model for a single fold with calibration.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        fold_id: Current fold number
        n_features: Number of features to select
        time_budget: Time limit for training (seconds)
        output_path: Directory for outputs
        target_col: Target column name
        
    Returns:
        predictions: Predicted labels
        probabilities: Predicted probabilities (calibrated)
        baseline_accuracy: Random forest baseline accuracy
        selected_features: List of selected feature names
        feature_scores: Feature importance scores
        explained_variance: PCA explained variance
        cumulative_variance: PCA cumulative variance
        calibration_metrics: Dictionary of calibration metrics
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    
    selected_features, feature_scores, explained_var, cumulative_var = select_features_pca(
        X_train[numeric_cols],
        n_features=n_features,
        fold_id=fold_id,
        save_visualizations=True,
        output_path=output_path
    )
    
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    min_class_size = min(pd.Series(y_train).value_counts())
    k_neighbors = min(5, min_class_size - 1) if min_class_size > 1 else 1
    
    try:
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
    except ValueError:
        X_resampled, y_resampled = X_train_scaled, y_train
    
    # Split for calibration
    split_idx = int(0.8 * len(X_resampled))
    X_cal = X_resampled[split_idx:]
    y_cal = y_resampled[split_idx:]
    X_train_final = X_resampled[:split_idx]
    y_train_final = y_resampled[:split_idx]
    
    df_train = pd.DataFrame(X_train_final, columns=selected_features)
    df_cal = pd.DataFrame(X_cal, columns=selected_features)
    df_test = pd.DataFrame(X_test_scaled, columns=selected_features)
    df_train[target_col] = y_train_final
    df_cal[target_col] = y_cal
    df_test[target_col] = y_test
    
    model_path = os.path.join(output_path, f"model_fold_{fold_id}")
    
    try:
        predictor = TabularPredictor(label=target_col, path=model_path, verbosity=0)
        predictor.fit(
            train_data=df_train,
            presets='best_quality',
            time_limit=time_budget,
            excluded_model_types=[]
        )
        
        # Calibration
        probs_cal = predictor.predict_proba(df_cal)
        temp_scaler = TemperatureScaling()
        temp_scaler.fit(probs_cal, y_cal)
        
        probs_uncalibrated = predictor.predict_proba(df_test)
        probs_calibrated = temp_scaler.transform(probs_uncalibrated)
        
        if isinstance(probs_calibrated, pd.DataFrame):
            predictions = probs_calibrated.idxmax(axis=1).values
        else:
            classes = sorted(y_test.unique())
            predictions = [classes[i] for i in np.argmax(probs_calibrated, axis=1)]
        
        ece_before = calculate_expected_calibration_error(y_test, probs_uncalibrated)
        ece_after = calculate_expected_calibration_error(y_test, probs_calibrated)
        
        calibration_metrics = {
            'temperature': temp_scaler.temperature,
            'ece_before': ece_before,
            'ece_after': ece_after,
            'ece_improvement': ece_before - ece_after
        }
        
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        
    except Exception:
        # Fallback to Random Forest
        rf_model = RandomForestClassifier(
            random_state=42, 
            class_weight='balanced', 
            n_estimators=100
        )
        rf_model.fit(X_resampled, y_resampled)
        predictions = rf_model.predict(X_test_scaled)
        probs_calibrated = pd.DataFrame(
            rf_model.predict_proba(X_test_scaled), 
            columns=rf_model.classes_
        )
        calibration_metrics = {
            'temperature': 1.0, 
            'ece_before': 0.0, 
            'ece_after': 0.0, 
            'ece_improvement': 0.0
        }
    
    # Baseline comparison
    rf_baseline = RandomForestClassifier(
        random_state=42, 
        class_weight='balanced', 
        n_estimators=100
    )
    rf_baseline.fit(X_resampled, y_resampled)
    baseline_accuracy = accuracy_score(y_test, rf_baseline.predict(X_test_scaled))
    
    return (
        predictions, 
        probs_calibrated, 
        baseline_accuracy, 
        selected_features,
        feature_scores, 
        explained_var, 
        cumulative_var, 
        calibration_metrics
    )


def generate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    output_path: str
) -> Tuple[Dict, pd.DataFrame]:
    """
    Generate and save confusion matrix and classification report.

    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10) * 100
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        annot_kws={"size": 18}
    )
    
    filepath = os.path.join(output_path, 'confusion_matrix_normalized.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    report_dict = classification_report(
        y_true, 
        y_pred, 
        labels=classes, 
        output_dict=True, 
        zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            report_df[['precision', 'recall', 'f1-score']],
            annot=True,
            cmap='YlOrRd',
            fmt='.2f',
            annot_kws={"size": 18}
        )
        
        filepath = os.path.join(output_path, 'classification_report_heatmap.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception:
        pass
    
    return report_dict, report_df


def save_results(
    df: pd.DataFrame,
    predictions: List,
    probabilities: List,
    classes: List[str],
    fold_metrics: List[Dict],
    final_metrics: Dict[str, float],
    report_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    input_path: str,
    output_path: str
) -> None:
    """
    Save all results to files.
    

    """
    results_df = df.copy()
    results_df['Predicted'] = predictions
    
    if probabilities and len(probabilities) == len(df):
        proba_df = pd.DataFrame(
            probabilities, 
            columns=[f'Prob_{cls}' for cls in classes]
        )
        for col in proba_df.columns:
            results_df[col] = proba_df[col]
    
    output_csv = input_path.replace('.csv', '_results.csv')
    results_df.to_csv(output_csv, index=False)
    
    metrics_df = pd.DataFrame(fold_metrics)
    summary_df = pd.DataFrame({
        'Metric': list(final_metrics.keys()),
        'Value': list(final_metrics.values())
    })
    
    excel_path = os.path.join(output_path, 'model_metrics.xlsx')
    
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            metrics_df.to_excel(writer, sheet_name='Fold_Metrics', index=False)
            summary_df.to_excel(writer, sheet_name='Summary_Metrics', index=False)
            report_df.to_excel(writer, sheet_name='Classification_Report')
            importance_df.to_excel(writer, sheet_name='Feature_Importance', index=False)
            consistency_df.to_excel(writer, sheet_name='Feature_Consistency', index=False)
    except ImportError:
        metrics_df.to_csv(os.path.join(output_path, 'fold_metrics.csv'), index=False)
        summary_df.to_csv(os.path.join(output_path, 'summary_metrics.csv'), index=False)
        report_df.to_csv(os.path.join(output_path, 'classification_report.csv'))
        importance_df.to_csv(os.path.join(output_path, 'feature_importance.csv'), index=False)
        consistency_df.to_csv(os.path.join(output_path, 'feature_consistency.csv'), index=False)


def main():
    """Execute the complete AutoML pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(42)
    
    df = load_data(INPUT_PATH, TARGET)
    
    min_class_size = df[TARGET].value_counts().min()
    if min_class_size < N_SPLITS:
        raise ValueError(
            f"Smallest class has {min_class_size} samples, "
            f"which is less than {N_SPLITS} splits"
        )
    
    exclude_cols = [col for col in EXCLUDE_COLUMNS if col in df.columns]
    X_full = df.drop(columns=exclude_cols + [TARGET])
    
    if 'DISTRICTE' in X_full.columns:
        X_full = pd.get_dummies(X_full, columns=['DISTRICTE'], prefix='DISTRICTE')
    
    y_full = df[TARGET]
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    fold_metrics = []
    all_selected_features = []
    all_feature_scores = []
    all_explained_variances = []
    all_cumulative_variances = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_full, y_full)):
        fold_num = fold_idx + 1
        
        X_train = X_full.iloc[train_idx]
        y_train = y_full.iloc[train_idx]
        X_test = X_full.iloc[test_idx]
        y_test = y_full.iloc[test_idx]
        
        (predictions, probabilities, baseline_acc, 
         selected_features, feature_scores, 
         explained_var, cumulative_var, 
         calibration_info) = train_fold(
            X_train, y_train, X_test, y_test,
            fold_num, N_FEATURES, TIME_LIMIT // N_SPLITS,
            OUTPUT_DIR, TARGET
        )
        
        all_selected_features.append(selected_features)
        all_feature_scores.append(feature_scores)
        all_explained_variances.append(explained_var)
        all_cumulative_variances.append(cumulative_var)
        
        acc = accuracy_score(y_test, predictions)
        f1_weighted = f1_score(y_test, predictions, average='weighted', zero_division=0)
        f1_macro = f1_score(y_test, predictions, average='macro', zero_division=0)
        kappa = cohen_kappa_score(y_test, predictions)
        
        fold_metrics.append({
            'Fold': fold_num,
            'Accuracy': acc,
            'F1_Weighted': f1_weighted,
            'F1_Macro': f1_macro,
            'Cohens_Kappa': kappa,
            'RF_Baseline_Accuracy': baseline_acc,
            'Temperature': calibration_info['temperature'],
            'ECE_Before': calibration_info['ece_before'],
            'ECE_After': calibration_info['ece_after'],
            'ECE_Improvement': calibration_info['ece_improvement'],
            'Selected_Features': ', '.join(selected_features)
        })
        
        all_predictions.extend(predictions)
        all_labels.extend(y_test)
        
        if isinstance(probabilities, pd.DataFrame):
            classes_ordered = sorted(y_full.unique())
            proba_aligned = []
            for _, row in probabilities.iterrows():
                aligned_row = [row.get(cls, 0.0) for cls in classes_ordered]
                proba_aligned.append(aligned_row)
            all_probabilities.extend(proba_aligned)
        else:
            all_probabilities.extend(probabilities.tolist())
    
    final_acc = accuracy_score(all_labels, all_predictions)
    final_f1_weighted = f1_score(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    final_f1_macro = f1_score(
        all_labels, all_predictions, average='macro', zero_division=0
    )
    final_kappa = cohen_kappa_score(all_labels, all_predictions)
    
    avg_temp = np.mean([m['Temperature'] for m in fold_metrics])
    avg_ece_before = np.mean([m['ECE_Before'] for m in fold_metrics])
    avg_ece_after = np.mean([m['ECE_After'] for m in fold_metrics])
    avg_improvement = np.mean([m['ECE_Improvement'] for m in fold_metrics])
    
    print(f"\n{'='*60}")
    print("CALIBRATION SUMMARY")
    print(f"{'='*60}")
    print(f"Average Temperature: {avg_temp:.3f}")
    print(f"Average ECE Before: {avg_ece_before:.3f}")
    print(f"Average ECE After: {avg_ece_after:.3f}")
    print(f"Average ECE Improvement: {avg_improvement:.3f}")
    
    if avg_improvement > 0.02:
        print("Calibration significantly improved probability reliability")
    elif avg_improvement > 0:
        print("Calibration slightly improved probability reliability")
    else:
        print("Probabilities were already well-calibrated")
    
    print(f"\n{'='*60}")
    print("FINAL CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {final_acc:.3f}")
    print(f"Weighted F1-Score: {final_f1_weighted:.3f}")
    print(f"Macro F1-Score: {final_f1_macro:.3f}")
    print(f"Cohen's Kappa: {final_kappa:.3f}")
    
    importance_df, consistency_df = aggregate_feature_analysis(
        all_feature_scores,
        all_selected_features,
        all_explained_variances,
        all_cumulative_variances,
        N_SPLITS,
        OUTPUT_DIR
    )
    
    classes = sorted(y_full.unique())
    report_dict, report_df = generate_confusion_matrix(
        all_labels, all_predictions, classes, OUTPUT_DIR
    )
    
    final_metrics = {
        'Accuracy': final_acc,
        'F1_Weighted': final_f1_weighted,
        'F1_Macro': final_f1_macro,
        'Cohens_Kappa': final_kappa,
        'Avg_Temperature': avg_temp,
        'Avg_ECE_Before': avg_ece_before,
        'Avg_ECE_After': avg_ece_after,
        'Avg_ECE_Improvement': avg_improvement
    }
    
    save_results(
        df, all_predictions, all_probabilities, classes,
        fold_metrics, final_metrics, report_df,
        importance_df, consistency_df,
        INPUT_PATH, OUTPUT_DIR
    )
    
    print(f"\n{'='*60}")
    print("FEATURE SELECTION SUMMARY")
    print(f"{'='*60}")
    most_consistent = consistency_df.head(5)
    for _, row in most_consistent.iterrows():
        consistency_pct = (row['Selection_Count'] / N_SPLITS) * 100
        print(
            f"{row['Feature']}: selected in {row['Selection_Count']}/{N_SPLITS} "
            f"folds ({consistency_pct:.1f}%)"
        )
    
    print(f"\n{'='*60}")
    print("OUTPUT FILES")
    print(f"{'='*60}")
    print(f"Results CSV: {INPUT_PATH.replace('.csv', '_results.csv')}")
    print(f"Metrics Excel: {os.path.join(OUTPUT_DIR, 'model_metrics.xlsx')}")
    print(f"Visualizations: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
