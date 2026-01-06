#!/usr/bin/env python3
"""
==============================================================================
MAIN PIPELINE: Rigorous Energy Forecasting & Optimization
==============================================================================
Orchestrates the end-to-end research protocol for h-step-ahead forecasting:

1. Data Loading & Leakage-Free Feature Engineering (with h-step target)
2. Chronological Splitting & TRUE Nested CV (with inner HP tuning)
3. Baseline Comparisons (Persistence, Seasonal Naive)
4. Final Model Training & Evaluation (with Uncertainty via Split Conformal)
5. Permutation Importance (on test set) with Negative Control Audit
6. Comprehensive Error Analysis (temporal breakdown, drift analysis)
7. Illustrative Toy Optimization (clearly labeled as demonstration)
8. Generation of ALL Publication-Ready Artifacts

CRITICAL DESIGN DECISIONS:
- Forecast horizon h is explicitly defined (default h=6 for 1-hour-ahead)
- At time t, all features use info from ≤t, target is y_{t+h}
- Split Conformal for uncertainty (documented as such, NOT Quantile Regression)
- Permutation importance on TEST set (not model-internal importance)
- Tree models (LightGBM) trained WITHOUT scaling (scale-invariant)

Author: Data Science Team
Date: November 2024 (Revised December 2024)
==============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import argparse
import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from scipy import stats

# Core Pipeline
import pipeline_core
from pipeline_core import (
    PipelineConfig, SafeLGBMRegressor, ConformalPredictor,
    calculate_metrics, calculate_uncertainty_metrics,
    calculate_error_breakdown, analyze_residual_drift,
    compute_persistence_baseline, compute_seasonal_naive_baseline,
    compute_permutation_importance, audit_negative_controls,
    run_nested_cv, diebold_mariano_test
)

# Optimization (labeled as demonstration only)
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Publication Plot Settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# ==============================================================================
# OUTPUT MANAGEMENT (run-scoped to avoid stale/mixed artifacts)
# ==============================================================================

OUTPUT_DIR = Path("outputs")

def set_output_dir(output_dir: str) -> None:
    """Set global output directory (run-scoped)."""
    global OUTPUT_DIR
    OUTPUT_DIR = Path(output_dir)

def outpath(*parts) -> str:
    """Join path parts under the current OUTPUT_DIR."""
    return str(OUTPUT_DIR.joinpath(*parts))

def ensure_dirs(output_dir: Path) -> None:
    """Create the standard output subfolders under a run-specific output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_importance").mkdir(parents=True, exist_ok=True)
    (output_dir / "error_analysis").mkdir(parents=True, exist_ok=True)
    (output_dir / "appendix").mkdir(parents=True, exist_ok=True)

def save_table(df, filename_base, caption=""):
    """Saves DataFrame as CSV and LaTeX under the current run OUTPUT_DIR."""
    csv_path = outpath(f"{filename_base}.csv")
    tex_path = outpath(f"{filename_base}.tex")

    df.to_csv(csv_path, index=False)

    with open(tex_path, 'w') as f:
        if caption:
            f.write(f"% {caption}\\n")
        f.write(df.to_latex(index=False, float_format="%.4f"))

    logger.info(f"Saved table: {filename_base}")



def generate_leakage_report(df, config):
    """Generates the Anti-Leakage Validation Report."""
    logger.info("Generating Leakage Check Report...")
    
    report_path = outpath('leakage_check_report.txt')
    with open(report_path, 'w') as f:
        f.write("LEAKAGE SANITY CHECK REPORT\n")
        f.write("===========================\n")
        f.write(f"Date: {datetime.now()}\n\n")
        
        f.write("1. FORECAST SETUP VERIFICATION\n")
        f.write(f"Forecast Horizon: h = {config.forecast_horizon_h} steps\n")
        f.write(f"For 10-min data: h={config.forecast_horizon_h} = {config.forecast_horizon_h * 10} minutes ahead\n\n")
        
        f.write("2. TARGET CONSTRUCTION\n")
        f.write("At time t, we predict target_ahead = y_{t+h}\n")
        f.write("All features use information from time <= t only.\n\n")
        
        target_col = config.target_col
        cols = [target_col, 'target_ahead']
        lag_cols = [c for c in df.columns if 'lag1' in c][:1]
        roll_cols = [c for c in df.columns if 'roll' in c][:2]
        display_cols = [c for c in cols + lag_cols + roll_cols if c in df.columns]
        
        mid = len(df) // 2
        sample = df[display_cols].iloc[mid:mid+10]
        
        f.write("Sample Data Window:\n")
        f.write(sample.to_string())
        f.write("\n\n")
        
        f.write("3. NEGATIVE CONTROL VARIABLES\n")
        f.write("rv1, rv2 are random variables used as negative controls.\n")
        f.write("If these show high importance, it indicates model issues.\n")
        
    logger.info(f"Saved: {report_path}")


def generate_split_report(train_df, test_df, config, sampling_interval_min=10):
    """Generates the Data Split Report."""
    report = {
        "forecast_horizon_h": config.forecast_horizon_h,
        "forecast_horizon_minutes": config.forecast_horizon_h * sampling_interval_min,
        "sampling_interval_minutes": sampling_interval_min,
        "prediction_setup": f"At time t, predict y_{{t+{config.forecast_horizon_h}}} using features from <=t",
        "train": {
            "start": str(train_df.index.min()),
            "end": str(train_df.index.max()),
            "samples": len(train_df)
        },
        "test": {
            "start": str(test_df.index.min()),
            "end": str(test_df.index.max()),
            "samples": len(test_df)
        },
        "chronological_split": True,
        "shuffled": False
    }
    
    with open(outpath('data_split_report.json'), "w") as f:
        json.dump(report, f, indent=4)
    logger.info(f"Saved: {outpath('data_split_report.json')}")


def plot_predictions(y_true, y_pred, index, model_name, h):
    """Plot predictions vs actual values."""
    subset = -144 * 7
    
    plt.figure(figsize=(12, 6))
    plt.plot(index[subset:], y_true[subset:], label='Actual', color='black', alpha=0.7, linewidth=1)
    plt.plot(index[subset:], y_pred[subset:], label='Predicted', color='blue', alpha=0.7, linewidth=1, linestyle='--')
    plt.title(f'{h}-Step-Ahead Predictions: {model_name} (Last 7 Days)')
    plt.ylabel('Energy (Wh)')
    plt.xlabel('Time')
    plt.legend()
    plt.savefig(outpath('fig_timeseries_pred_vs_true.png'))
    plt.close()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.1, s=2)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual (Wh)')
    plt.ylabel('Predicted (Wh)')
    plt.title(f'Actual vs Predicted: {model_name} ({h}-Step-Ahead)')
    plt.legend()
    plt.savefig(outpath('fig_scatter_pred_vs_true.png'))
    plt.close()


def plot_residuals(y_true, y_pred, index, h):
    """Plot residual analysis."""
    residuals = y_true - y_pred
    
    plt.figure(figsize=(12, 6))
    plt.plot(index, residuals, alpha=0.5, linewidth=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'Residuals Over Time ({h}-Step-Ahead)')
    plt.ylabel('Error (Wh)')
    plt.xlabel('Time')
    
    rolling_mean = pd.Series(residuals, index=index).rolling(144).mean()
    plt.plot(index, rolling_mean, color='orange', linewidth=2, label='24h Rolling Mean')
    plt.legend()
    plt.savefig(outpath('fig_residuals_over_time.png'))
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(residuals, kde=True, bins=50, ax=axes[0])
    axes[0].axvline(0, color='red', linestyle='--')
    axes[0].set_title('Residual Distribution')
    axes[0].set_xlabel('Residual (Wh)')
    
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (Normality Check)')
    
    plt.tight_layout()
    plt.savefig(outpath('fig_residual_distribution.png'))
    plt.close()


def plot_uncertainty(y_true, y_lower, y_upper, index, alpha, h):
    """Plot prediction intervals."""
    subset = slice(-144*7, -144*6)
    idx_sub = index[subset]
    
    plt.figure(figsize=(12, 6))
    plt.fill_between(idx_sub, y_lower[subset], y_upper[subset], 
                     color='blue', alpha=0.2, label=f'{int((1-alpha)*100)}% Prediction Interval')
    plt.plot(idx_sub, y_true[subset], color='black', label='Actual', linewidth=1.5)
    plt.title(f'Prediction Intervals ({h}-Step-Ahead, Split Conformal)')
    plt.ylabel('Energy (Wh)')
    plt.xlabel('Time')
    plt.legend()
    plt.savefig(outpath('fig_prediction_intervals_sample_week.png'))
    plt.close()


def plot_error_breakdown(error_df, h):
    """Plot error breakdown by temporal categories."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    hour_data = error_df[error_df['Category'] == 'Hour'].copy()
    hour_data['Value'] = hour_data['Value'].astype(int)
    hour_data = hour_data.sort_values('Value')
    
    axes[0, 0].bar(hour_data['Value'], hour_data['RMSE'], color='steelblue', alpha=0.7)
    axes[0, 0].set_xlabel('Hour of Day')
    axes[0, 0].set_ylabel('RMSE (Wh)')
    axes[0, 0].set_title('Error by Hour of Day')
    axes[0, 0].set_xticks(range(0, 24, 3))
    
    dow_data = error_df[error_df['Category'] == 'DayOfWeek']
    day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_data = dow_data.set_index('Value').loc[day_order].reset_index()
    
    colors = ['steelblue'] * 5 + ['coral'] * 2
    axes[0, 1].bar(dow_data['Value'], dow_data['RMSE'], color=colors, alpha=0.7)
    axes[0, 1].set_xlabel('Day of Week')
    axes[0, 1].set_ylabel('RMSE (Wh)')
    axes[0, 1].set_title('Error by Day of Week')
    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
    
    wknd_data = error_df[error_df['Category'] == 'WeekendFlag']
    axes[1, 0].bar(wknd_data['Value'], wknd_data['RMSE'], color=['steelblue', 'coral'], alpha=0.7)
    axes[1, 0].set_xlabel('Period')
    axes[1, 0].set_ylabel('RMSE (Wh)')
    axes[1, 0].set_title('Error: Weekday vs Weekend')
    
    axes[1, 1].bar(hour_data['Value'], hour_data['MBE'], 
                   color=['green' if x < 0 else 'red' for x in hour_data['MBE']], alpha=0.7)
    axes[1, 1].axhline(0, color='black', linestyle='--')
    axes[1, 1].set_xlabel('Hour of Day')
    axes[1, 1].set_ylabel('Mean Bias Error (Wh)')
    axes[1, 1].set_title('Prediction Bias by Hour')
    axes[1, 1].set_xticks(range(0, 24, 3))
    
    plt.suptitle(f'{h}-Step-Ahead Forecast Error Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(outpath('error_analysis', 'fig_error_breakdown.png'))
    plt.close()


def plot_feature_importance_with_controls(importance_df, negative_control_cols, h):
    """Plot feature importance with negative controls highlighted."""
    plt.figure(figsize=(10, 12))
    
    top_df = importance_df.head(25).copy()
    
    colors = []
    for feat in top_df['Feature']:
        if any(ctrl in feat for ctrl in negative_control_cols):
            colors.append('red')
        else:
            colors.append('steelblue')
    
    plt.barh(range(len(top_df)), top_df['Importance'], color=colors, alpha=0.7)
    
    if 'Importance_Std' in top_df.columns:
        plt.errorbar(top_df['Importance'], range(len(top_df)), 
                    xerr=top_df['Importance_Std'], fmt='none', color='black', capsize=3)
    
    plt.yticks(range(len(top_df)), top_df['Feature'])
    plt.xlabel('Permutation Importance (increase in MSE)')
    plt.title(f'Feature Importance ({h}-Step-Ahead)\nRed = Negative Controls')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(outpath('fig_feature_importance.png'))
    plt.close()


def plot_baseline_comparison(metrics_dict, h):
    """Plot comparison with baseline models."""
    models = list(metrics_dict.keys())
    rmse_vals = [metrics_dict[m]['RMSE'] for m in models]
    mae_vals = [metrics_dict[m]['MAE'] for m in models]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = []
    for m in models:
        if 'Persistence' in m or 'Naive' in m or 'Seasonal' in m:
            colors.append('gray')
        else:
            colors.append('steelblue')
    
    axes[0].bar(models, rmse_vals, color=colors, alpha=0.7)
    axes[0].set_ylabel('RMSE (Wh)')
    axes[0].set_title(f'RMSE Comparison ({h}-Step-Ahead)')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    axes[1].bar(models, mae_vals, color=colors, alpha=0.7)
    axes[1].set_ylabel('MAE (Wh)')
    axes[1].set_title(f'MAE Comparison ({h}-Step-Ahead)')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(outpath('fig_baseline_comparison.png'))
    plt.close()


def generate_hourly_heatmap(df, target_col, h):
    """Generates a 4-week hourly heatmap from the test set."""
    logger.info("Generating 4-week hourly heatmap (Deliverable A)...")

    # Select first 28 days of data
    df_subset = df.iloc[:28*144].copy() # 28 days * 144 (10-min intervals)

    # Resample to hourly mean
    df_hourly = df_subset[[target_col]].resample('H').mean()

    # Create pivot table: Day (y-axis) vs Hour (x-axis)
    df_hourly['day'] = df_hourly.index.date
    df_hourly['hour'] = df_hourly.index.hour

    pivot_table = df_hourly.pivot(index='day', columns='hour', values=target_col)

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, cmap='viridis', cbar_kws={'label': 'Energy (Wh)'})
    plt.title(f'Hourly Energy Consumption Heatmap (First 4 Weeks of Test Set)')
    plt.ylabel('Date')
    plt.xlabel('Hour of Day')
    plt.tight_layout()
    plt.savefig(outpath('fig_hourly_heatmap.png'))
    plt.close()


def run_scenario_analysis(config, pipe_config, df_eng, metrics_main):
    """
    Runs additional scenarios for the results table (Deliverable C).
    Scenarios:
    1. All features (metrics_main)
    2. No Lights (computed in Phase 2, passed if available, otherwise computed here)
    3. No Weather (Exclude T_out, Press, RH_out, Windspeed, Visibility, Tdewpoint)
    4. Only Weather + Time (Exclude indoor sensors T*, RH_*, and lights)
    """
    logger.info("Running Scenario Analysis (Deliverable C)...")

    # Define excluded columns for "No Weather"
    # Must exclude raw weather columns AND derived features that depend on them
    weather_cols = ['T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint']
    weather_derived = ['DeltaT', 'DeltaT_abs'] # derived using T_out

    # Define excluded columns for "Only Weather + Time"
    # Must exclude all indoor sensors, lights, and indoor-derived aggregates
    indoor_cols = []
    for i in range(1, 10):
        indoor_cols.append(f'T{i}')
        indoor_cols.append(f'RH_{i}')

    # Indoor derived features
    indoor_derived = ['T_indoor_avg', 'T_indoor_std', 'RH_indoor_avg']

    # Negative controls to exclude from ALL scenarios (Strict Match)
    # We filter these dynamically in the loop to be robust

    scenarios = [
        {'name': 'All Features', 'exclude': [], 'uncertainty_source': metrics_main},
        {'name': 'No Lights', 'exclude': ['lights']},
        {'name': 'No Weather', 'exclude': weather_cols + weather_derived},
        {'name': 'Only Weather + Time', 'exclude': indoor_cols + indoor_derived + weather_derived + ['lights']}
    ]

    results = []

    X_train, X_test, y_train, y_test, all_feature_cols = pipeline_core.get_chronological_split(
        df_eng, pipe_config
    )

    # Train/Test logic
    for scen in scenarios:
        row = {'Scenario': scen['name']}

        # Determine features to use
        # Start with all features
        features_to_use = [f for f in all_feature_cols]

        # Remove excluded features
        exclude_list = scen['exclude']
        features_to_use = [f for f in features_to_use if f not in exclude_list]

        # ALWAYS remove negative controls for scientific scenarios
        # Robust filtering: exact match for rv1/rv2, startswith for negctrl
        features_to_use = [
            f for f in features_to_use
            if f not in ['rv1', 'rv2'] and not f.startswith('negctrl')
        ]

        logger.info(f"  Scenario: {scen['name']} (Features: {len(features_to_use)})")

        # Prepare Data
        # Map feature names to indices in X_train/X_test
        # Wait, X_train is numpy array. Need to use DataFrame or map indices.
        # Let's rebuild DataFrames
        X_train_df = pd.DataFrame(X_train, columns=all_feature_cols)
        X_test_df = pd.DataFrame(X_test, columns=all_feature_cols)

        X_train_sub = X_train_df[features_to_use]
        X_test_sub = X_test_df[features_to_use]

        # Train Model
        model = SafeLGBMRegressor(**config['models']['lightgbm'], random_state=42, verbose=-1)
        model.fit(X_train_sub, y_train)
        y_pred = model.predict(X_test_sub)

        # Metrics
        m = calculate_metrics(y_test, y_pred)

        # Add Uncertainty if source provided (e.g. for All Features)
        picp, mpiw = np.nan, np.nan
        if 'uncertainty_source' in scen:
             source = scen['uncertainty_source']
             picp = source.get('PICP', np.nan)
             mpiw = source.get('MPIW', np.nan)

             if np.isnan(picp):
                 logger.warning(f"Uncertainty metrics missing for scenario '{scen['name']}' (likely running in --quick mode).")

        row.update({
            'RMSE': m['RMSE'], 'MAE': m['MAE'], 'R2': m['R2'],
            'PICP': picp, 'MPIW': mpiw
        })
        results.append(row)

    return pd.DataFrame(results)


def run_experiment(config, pipe_config, include_lights=True, perform_full_analysis=False):
    """Run the main forecasting experiment."""
    scenario = "With Lights" if include_lights else "Without Lights"
    logger.info(f"Running Experiment: {scenario}")
    logger.info(f"Forecast Horizon: h = {pipe_config.forecast_horizon_h} steps")
    
    pipe_config.include_lights = include_lights
    
    df_raw = pipeline_core.load_data(config['paths']['data'])
    df_eng = pipeline_core.create_physics_features(df_raw, pipe_config)
    
    pipeline_core.check_leakage(df_eng, pipe_config.target_col, pipe_config.forecast_horizon_h)
    
    X_train, X_test, y_train, y_test, feature_cols = pipeline_core.get_chronological_split(
        df_eng, pipe_config
    )
    
    test_size = int(len(df_eng) * pipe_config.test_size_percent)
    train_size = len(df_eng) - test_size
    test_index = df_eng.index[train_size:]

    # Assumption audit: clarify whether current load y_t is being used as a predictor (directly or via diffs)
    current_load_features_present = [
        c for c in feature_cols
        if c.endswith('_diff1') or c.endswith('_diff6') or c.endswith('_lag0')
    ]
    assumption_audit = {
        "scenario": scenario,
        "forecast_horizon_h": int(pipe_config.forecast_horizon_h),
        "current_load_available": bool(pipe_config.current_load_available),
        "current_load_features_present": current_load_features_present,
        "notes": "If current_load_available=False, features that depend on y_t (diff/lag0) should be absent."
    }
    with open(outpath('assumption_audit.json'), 'w') as f:
        json.dump(assumption_audit, f, indent=4)
    
    if perform_full_analysis:
        train_df = df_eng.iloc[:train_size]
        test_df = df_eng.iloc[train_size:]
        generate_split_report(train_df, test_df, pipe_config)
        generate_leakage_report(df_eng, pipe_config)

    # CRITICAL FIX: LightGBM is a tree model - NO SCALING NEEDED
    # Tree models are scale-invariant; scaling adds complexity and potential mismatch issues
    # This eliminates the permutation importance target-scale mismatch entirely
    
    logger.info("Computing baseline predictions...")
    
    y_pred_persistence = compute_persistence_baseline(
        y_train, y_test, h=pipe_config.forecast_horizon_h,
        current_load_available=pipe_config.current_load_available
    )
    
    y_pred_seasonal_daily = compute_seasonal_naive_baseline(
        y_train, y_test, seasonal_period=144
    )
    
    if len(y_train) >= 1008:
        y_pred_seasonal_weekly = compute_seasonal_naive_baseline(
            y_train, y_test, seasonal_period=1008
        )
    else:
        y_pred_seasonal_weekly = None
    
    logger.info("Training main model (LightGBM)...")
    # Train on ORIGINAL scale - no scaling for tree models
    # Use DataFrames with feature names to avoid sklearn warnings
    X_train_df = pd.DataFrame(X_train, columns=feature_cols)
    X_test_df = pd.DataFrame(X_test, columns=feature_cols)
    
    model = SafeLGBMRegressor(**config['models']['lightgbm'], random_state=42, verbose=-1)
    model.fit(X_train_df, y_train)
    
    y_pred = model.predict(X_test_df)
    
    metrics = calculate_metrics(y_test, y_pred)
    metrics['Model'] = 'LightGBM'
    metrics['Scenario'] = scenario
    metrics['Forecast_Horizon'] = pipe_config.forecast_horizon_h
    
    # FIX: Add consistent metadata to ALL baseline rows
    baseline_metrics = {}
    baseline_metrics['LightGBM'] = metrics.copy()
    
    pers_metrics = calculate_metrics(y_test, y_pred_persistence)
    pers_metrics['Model'] = 'Persistence'
    pers_metrics['Scenario'] = scenario
    pers_metrics['Forecast_Horizon'] = pipe_config.forecast_horizon_h
    baseline_metrics['Persistence'] = pers_metrics
    
    seas_daily_metrics = calculate_metrics(y_test, y_pred_seasonal_daily)
    seas_daily_metrics['Model'] = 'Seasonal_Daily'
    seas_daily_metrics['Scenario'] = scenario
    seas_daily_metrics['Forecast_Horizon'] = pipe_config.forecast_horizon_h
    baseline_metrics['Seasonal_Daily'] = seas_daily_metrics
    
    if y_pred_seasonal_weekly is not None:
        seas_weekly_metrics = calculate_metrics(y_test, y_pred_seasonal_weekly)
        seas_weekly_metrics['Model'] = 'Seasonal_Weekly'
        seas_weekly_metrics['Scenario'] = scenario
        seas_weekly_metrics['Forecast_Horizon'] = pipe_config.forecast_horizon_h
        baseline_metrics['Seasonal_Weekly'] = seas_weekly_metrics
    
    dm_stat, dm_pval = diebold_mariano_test(y_test, y_pred, y_pred_persistence, 
                                             h=pipe_config.forecast_horizon_h)
    metrics['DM_vs_Persistence_pval'] = dm_pval
    
    dm_stat, dm_pval = diebold_mariano_test(y_test, y_pred, y_pred_seasonal_daily,
                                             h=pipe_config.forecast_horizon_h)
    metrics['DM_vs_Seasonal_pval'] = dm_pval
    
    if perform_full_analysis:
        logger.info("Performing full analysis...")
        
        logger.info("Computing prediction intervals (Split Conformal)...")
        
        # Conformal prediction on ORIGINAL scale (no scaling for tree models)
        # Use DataFrames with feature names for consistency
        calib_frac = float(config.get('uncertainty', {}).get('calib_fraction', 0.2))
        calib_size = int(len(X_train_df) * calib_frac)
        train_proper_size = len(X_train_df) - calib_size
        
        X_t_prop = X_train_df.iloc[:train_proper_size]
        X_cal = X_train_df.iloc[train_proper_size:]
        y_t_prop, y_cal = y_train[:train_proper_size], y_train[train_proper_size:]
        
        alpha = float(config.get('uncertainty', {}).get('alpha', 0.1))
        base_model = SafeLGBMRegressor(**config['models']['lightgbm'], random_state=42, verbose=-1)
        cp = ConformalPredictor(base_model, alpha=alpha)
        cp.fit(X_t_prop, y_t_prop, X_cal, y_cal)
        
        y_pred_cp, y_lower, y_upper = cp.predict(X_test_df)

        # --- CONSISTENCY & AUDIT ---
        # From this point onward, use the SAME estimator/predictions that produced the conformal intervals.
        y_pred_used = y_pred_cp
        model_used = cp.model

        # Diagnostic audit: compare to the full-train point model (do NOT report unless methodology is aligned)
        rmse_point = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
        rmse_conformal = float(np.sqrt(np.mean((y_test - y_pred_used) ** 2)))
        current_load_features = [
            c for c in feature_cols
            if c.endswith('_diff1') or c.endswith('_diff6') or c.endswith('_lag0')
        ]
        audit = {
            "scenario": scenario,
            "forecast_horizon_h": int(pipe_config.forecast_horizon_h),
            "current_load_available": bool(pipe_config.current_load_available),
            "current_load_features_present": current_load_features,
            "rmse_point_model_full_train": rmse_point,
            "rmse_conformal_base_model": rmse_conformal,
            "delta_rmse": float(rmse_point - rmse_conformal)
        }
        with open(outpath('error_analysis', 'consistency_audit.json'), 'w') as f:
            json.dump(audit, f, indent=4)

        logger.info(
            f"Consistency audit saved to {outpath('error_analysis','consistency_audit.json')}. "
            f"RMSE(full-train point)={rmse_point:.4f}, RMSE(conformal base)={rmse_conformal:.4f}"
        )

        # Ensure reported point metrics match the estimator used for intervals (prevents overclaiming)
        point_metrics_used = calculate_metrics(y_test, y_pred_used)
        for k, v in point_metrics_used.items():
            metrics[k] = v
        baseline_metrics['LightGBM'] = metrics.copy()

        unc_metrics = calculate_uncertainty_metrics(y_test, y_lower, y_upper, alpha=alpha)
        metrics.update(unc_metrics)
        
        logger.info("Computing error breakdown...")
        error_breakdown = calculate_error_breakdown(y_test, y_pred_used, test_index)
        error_breakdown.to_csv(outpath('error_analysis', 'error_breakdown.csv'), index=False)
        
        drift_analysis = analyze_residual_drift(y_test - y_pred_used, test_index)
        with open(outpath('error_analysis', 'drift_analysis.json'), 'w') as f:
            json.dump(drift_analysis, f, indent=4)
        
        logger.info("Computing permutation importance on test set...")
        # Model trained on original scale -> use original y_test
        # Use DataFrame to avoid feature name warnings
        importance_df = compute_permutation_importance(
            model_used,
            X_test_df,
            y_test,
            feature_cols,
            n_repeats=int(config.get('settings', {}).get('perm_importance_repeats', 10)),
            random_state=42
        )
        importance_df.to_csv(outpath('feature_importance.csv'), index=False)
        
        logger.info("Auditing negative control variables...")
        audit_results = audit_negative_controls(
            importance_df, pipe_config.negative_control_cols
        )
        
        with open(outpath('negative_control_audit.json'), 'w') as f:
            json.dump(audit_results, f, indent=4, default=str)
        
        if not audit_results['passed']:
            logger.warning("NEGATIVE CONTROL AUDIT FAILED!")
            for warning in audit_results['warnings']:
                logger.warning(warning)
        else:
            logger.info("Negative control audit PASSED")
        
        h = pipe_config.forecast_horizon_h
        
        plot_predictions(y_test, y_pred_used, test_index, 'LightGBM', h)
        plot_residuals(y_test, y_pred_used, test_index, h)
        plot_uncertainty(y_test, y_lower, y_upper, test_index, alpha, h)
        plot_error_breakdown(error_breakdown, h)
        plot_baseline_comparison(baseline_metrics, h)
        plot_feature_importance_with_controls(
            importance_df, pipe_config.negative_control_cols, h
        )
    
    return metrics, baseline_metrics


def run_nested_cv_experiment(config, pipe_config):
    """Run TRUE nested cross-validation with inner-loop hyperparameter tuning."""
    logger.info("Running TRUE Nested Cross-Validation...")
    
    df_raw = pipeline_core.load_data(config['paths']['data'])
    df_eng = pipeline_core.create_physics_features(df_raw, pipe_config)
    
    cv_results = run_nested_cv(df_eng, pipe_config)
    
    suffix = "with_lights" if pipe_config.include_lights else "no_lights"
    cv_results.to_csv(outpath(f'nested_cv_results_{suffix}.csv'), index=False)
    
    summary = cv_results.groupby('Model').agg({
        'Test_RMSE': ['mean', 'std'],
        'Test_MAE': ['mean', 'std'],
        'Test_R2': ['mean', 'std'],
        'Test_NRMSE': ['mean', 'std']
    }).round(4)
    
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary.to_csv(outpath(f'nested_cv_summary_{suffix}.csv'), index=False)
    
    logger.info(f"Nested CV completed. Results saved.")
    
    return cv_results, summary


def run_toy_optimization_demonstration(baseline_daily_energy_kwh):
    """DEMONSTRATION-ONLY: Illustrative Toy Optimization."""
    logger.info("Running TOY Optimization (DEMONSTRATION ONLY)...")
    logger.warning("This optimization is for demonstration only. NOT validated.")
    
    class ToyProblem(Problem):
        def __init__(self, baseline):
            super().__init__(n_var=8, n_obj=2, xl=18.0, xu=26.0)
            self.baseline = baseline
            
        def _evaluate(self, X, out, *args, **kwargs):
            avg_setpoint = np.mean(X, axis=1)
            energy_factor = 1.0 + (avg_setpoint - 21.5) * 0.05
            energy_factor = np.clip(energy_factor, 0.7, 1.3)
            energy = self.baseline * energy_factor
            
            discomfort = np.mean(np.abs(X - 21.0), axis=1) * 0.2
            
            out["F"] = np.column_stack([energy, discomfort])
    
    problem = ToyProblem(baseline_daily_energy_kwh)
    algorithm = NSGA2(
        pop_size=100,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    res = minimize(problem, algorithm, ('n_gen', 50), seed=1, verbose=False)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(res.F[:, 0], res.F[:, 1], c='blue', alpha=0.6, s=30)
    plt.xlabel('Proxy Energy (kWh/day)')
    plt.ylabel('Proxy Discomfort Index')
    plt.title('DEMONSTRATION: Illustrative Pareto Front\n(Proxy-Based, NOT Validated)')
    
    plt.figtext(0.5, 0.02, 
                "DEMONSTRATION ONLY: Uses proxy objectives, not calibrated physics.",
                ha='center', fontsize=9, style='italic', color='red')
    
    plt.savefig(outpath('appendix', 'fig_toy_pareto_demonstration.png'))
    plt.close()
    
    pareto_df = pd.DataFrame({
        'Proxy_Energy_kWh': res.F[:, 0],
        'Proxy_Discomfort': res.F[:, 1],
        **{f'Setpoint_Hour_{i+1}': res.X[:, i] for i in range(8)}
    })
    pareto_df.to_csv(outpath('appendix', 'toy_pareto_solutions.csv'), index=False)
    
    logger.info(f"Toy optimization saved to {outpath('appendix')}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Energy Forecasting Pipeline")
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--forecast-horizon', type=int, default=6)
    parser.add_argument('--skip-nested-cv', action='store_true')
    parser.add_argument('--skip-optimization', action='store_true')
    parser.add_argument('--no-current-load', action='store_true', help='Assume current load y_t is NOT available at issuance time')
    parser.add_argument('--quick', action='store_true', help='Smoke test: skip full analysis (conformal/importance/plots) in Phase 1')
    args = parser.parse_args()
    
    config = load_config(args.config)

    # Run-scoped output directory to avoid stale/mixed artifacts
    base_outputs = config.get('paths', {}).get('outputs', 'outputs')
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(base_outputs) / f"run_{run_id}_h{args.forecast_horizon}"
    set_output_dir(str(run_dir))
    ensure_dirs(run_dir)

    # Persist run metadata for traceability
    run_meta = {
        'run_id': run_id,
        'forecast_horizon_h': int(args.forecast_horizon),
        'skip_nested_cv': bool(args.skip_nested_cv),
        'skip_optimization': bool(args.skip_optimization),
        'quick': bool(args.quick),
        'timestamp_utc': datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
    }
    with open(outpath('run_metadata.json'), 'w') as f:
        json.dump(run_meta, f, indent=4)
    
    settings = config.get('settings', {})
    fe = config.get('feature_engineering', {}) or {}

    pipe_config = PipelineConfig(
        target_col=settings['target_col'],
        time_col=settings.get('time_col', 'date'),
        test_size_percent=float(settings.get('test_size_percent', 0.25)),
        random_seed=int(settings.get('random_seed', 42)),
        forecast_horizon_h=int(args.forecast_horizon),
        include_lights=bool(settings.get('include_lights', True)),
        lags=[int(x) for x in fe.get('lags', [1, 2, 3, 6, 12, 24, 36])],
        rolling_windows=[int(x) for x in fe.get('rolling_windows', [6, 12, 24])],
        n_outer_folds=int(settings.get('n_outer_folds', 5)),
        n_inner_folds=int(settings.get('n_inner_folds', 3)),
        n_iter_random_search=int(settings.get('n_iter_random_search', 20)),
        use_full_hyperparameter_search=True,
        current_load_available=(not args.no_current_load) and bool(
            config.get('assumptions', {}).get('current_load_available', True)
        )
    )

    # Extend run metadata with the *effective* configuration (single source of truth).
    try:
        run_meta.update({
            'time_col': pipe_config.time_col,
            'random_seed': pipe_config.random_seed,
            'lags': pipe_config.lags,
            'rolling_windows': pipe_config.rolling_windows,
            'current_load_available': pipe_config.current_load_available,
            'uncertainty': config.get('uncertainty', {}),
            'config_path': os.path.abspath(args.config)
        })
        with open(outpath('run_metadata.json'), 'w') as f:
            json.dump(run_meta, f, indent=4)
    except Exception as e:
        logger.warning(f"Could not update run_metadata.json with effective config: {e}")

    
    logger.info("=" * 60)
    logger.info("ENERGY FORECASTING PIPELINE")
    logger.info(f"Forecast Horizon: h={pipe_config.forecast_horizon_h} steps")
    logger.info("=" * 60)
    
    logger.info("\nPHASE 1: Main Experiment (With Lights)")
    metrics_main, baseline_metrics = run_experiment(
        config, pipe_config, include_lights=True, perform_full_analysis=(not args.quick)
    )
    
    logger.info("\nPHASE 2: Comparison Experiment (Without Lights)")
    metrics_no_lights, _ = run_experiment(
        config, pipe_config, include_lights=False, perform_full_analysis=False
    )
    
    if not args.skip_nested_cv:
        logger.info("\nPHASE 3: Nested Cross-Validation")
        pipe_config.include_lights = True
        cv_results_lights, _ = run_nested_cv_experiment(config, pipe_config)
        
        pipe_config.include_lights = False
        cv_results_no_lights, _ = run_nested_cv_experiment(config, pipe_config)
    
    # Deliverable A: 4-week hourly heatmap
    # Generate regardless of --quick mode (it is lightweight)
    df_raw = pipeline_core.load_data(config['paths']['data'])
    df_eng = pipeline_core.create_physics_features(df_raw, pipe_config)
    _, _, _, _, _ = pipeline_core.get_chronological_split(df_eng, pipe_config) # Just to get indices
    test_size = int(len(df_eng) * pipe_config.test_size_percent)
    test_df = df_eng.iloc[-test_size:]
    generate_hourly_heatmap(test_df, pipe_config.target_col, pipe_config.forecast_horizon_h)

    # Deliverable C: Scenario Table
    # Need the full DF again for scenario analysis
    df_raw = pipeline_core.load_data(config['paths']['data'])
    # Ensure pipe_config.include_lights is True for the "All Features" base
    pipe_config.include_lights = True
    df_eng = pipeline_core.create_physics_features(df_raw, pipe_config)

    # Run Scenario Analysis
    # We pass metrics_main (All features) to avoid re-running it.
    # For "No Lights", we have metrics_no_lights, but run_scenario_analysis expects to run logic.
    # Let's just update the list in run_scenario_analysis to accept pre-computed.

    df_scenarios = run_scenario_analysis(config, pipe_config, df_eng, metrics_main)

    # Sort or reorder
    order = ['All Features', 'No Lights', 'No Weather', 'Only Weather + Time']
    df_scenarios['Scenario'] = pd.Categorical(df_scenarios['Scenario'], categories=order, ordered=True)
    df_scenarios = df_scenarios.sort_values('Scenario')

    save_table(df_scenarios, f'results_scenarios_h{pipe_config.forecast_horizon_h}')

    logger.info("\nPHASE 4: Generating Summary Tables")
    
    df_ts = pd.DataFrame([metrics_main])
    save_table(df_ts, 'results_timeseries')
    
    baseline_rows = []
    for model_name, m in baseline_metrics.items():
        row = {'Model': model_name}
        row.update(m)
        baseline_rows.append(row)
    df_baselines = pd.DataFrame(baseline_rows)
    save_table(df_baselines, 'baseline_comparison')
    
    df_comp = pd.DataFrame([
        {'Scenario': 'With Lights', 
         'RMSE': metrics_main['RMSE'], 
         'MAE': metrics_main['MAE'], 
         'R2': metrics_main['R2'],
         'NRMSE': metrics_main['NRMSE'],
         'SMAPE': metrics_main['SMAPE']},
        {'Scenario': 'Without Lights', 
         'RMSE': metrics_no_lights['RMSE'], 
         'MAE': metrics_no_lights['MAE'], 
         'R2': metrics_no_lights['R2'],
         'NRMSE': metrics_no_lights['NRMSE'],
         'SMAPE': metrics_no_lights['SMAPE']}
    ])
    save_table(df_comp, 'results_with_vs_without_lights')
    
    if not args.skip_optimization:
        logger.info("\nPHASE 5: Toy Optimization (DEMONSTRATION)")
        df_tmp = pd.read_csv(config['paths']['data'])
        baseline_est = df_tmp['Appliances'].mean() * 144 / 1000
        run_toy_optimization_demonstration(baseline_est)
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nBest Model RMSE: {metrics_main['RMSE']:.2f} Wh")
    logger.info(f"Best Model R2: {metrics_main['R2']:.4f}")
    if 'PICP' in metrics_main:
        logger.info(f"90% PI Coverage: {metrics_main['PICP']:.1f}%")
    logger.info(f"\nAll artifacts saved to {OUTPUT_DIR}/")
