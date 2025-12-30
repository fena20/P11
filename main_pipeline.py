#!/usr/bin/env python3
"""
==============================================================================
MAIN PIPELINE: Rigorous Energy Forecasting & Replication
==============================================================================
Orchestrates the end-to-end research protocol:

Task A: Strict Replication of Candanedo et al. (2017)
        (h=0, Stratified Random Split, Paper Features)
Task B: Methodological Correction
        (h=0, Chronological Split, Paper Features)
Task C: Forecasting Extension
        (h=6, Chronological, Strong Baselines, Uncertainty, Feature Availability)

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
import sklearn

# Core Pipeline
import pipeline_core
from pipeline_core import (
    PipelineConfig, ConformalPredictor, create_pipeline,
    calculate_metrics, calculate_uncertainty_metrics,
    calculate_error_breakdown, analyze_residual_drift,
    compute_persistence_baseline, compute_seasonal_naive_baseline,
    compute_time_of_week_mean_baseline,
    compute_permutation_importance, audit_negative_controls,
    get_paper_mode_features, get_data_split, create_strict_paper_features,
    tune_model
)

# Optimization (Demonstration)
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Output Directory (Global)
OUTPUT_DIR = Path("outputs")

# Enable pandas output for sklearn transformers to preserve feature names where possible
sklearn.set_config(transform_output="pandas")

def set_output_dir(output_dir: str) -> None:
    global OUTPUT_DIR
    OUTPUT_DIR = Path(output_dir)

def outpath(*parts) -> str:
    return str(OUTPUT_DIR.joinpath(*parts))

def ensure_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_importance").mkdir(parents=True, exist_ok=True)
    (output_dir / "error_analysis").mkdir(parents=True, exist_ok=True)
    (output_dir / "replication").mkdir(parents=True, exist_ok=True)
    (output_dir / "appendix").mkdir(parents=True, exist_ok=True)

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# ==============================================================================
# PLOTTING HELPERS
# ==============================================================================

def plot_predictions(y_true, y_pred, index, model_name, h):
    subset = slice(-144*7, None)
    plt.figure(figsize=(12, 6))
    try:
        idx_sub = index[subset]
        y_true_sub = y_true[subset]
        y_pred_sub = y_pred[subset]
    except Exception:
        # Fallback if index slicing fails
        idx_sub = range(len(y_true))[-144*7:]
        y_true_sub = y_true[-144*7:]
        y_pred_sub = y_pred[-144*7:]

    plt.plot(idx_sub, y_true_sub, label='Actual', color='black', alpha=0.7)
    plt.plot(idx_sub, y_pred_sub, label='Predicted', color='blue', alpha=0.7, linestyle='--')
    plt.title(f'{h}-Step-Ahead Predictions: {model_name} (Last 7 Days)')
    plt.ylabel('Energy (Wh)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath(f'fig_pred_vs_true_h{h}.png'))
    plt.close()

def plot_uncertainty(y_true, y_lower, y_upper, index, alpha, h):
    subset = slice(-144*7, -144*6) # One day zoom
    try:
        idx_sub = index[subset]
        y_lower_sub = y_lower[subset]
        y_upper_sub = y_upper[subset]
        y_true_sub = y_true[subset]
    except Exception:
        return # Skip if index issues

    plt.figure(figsize=(12, 6))
    plt.fill_between(idx_sub, y_lower_sub, y_upper_sub, color='blue', alpha=0.2, label='PI')
    plt.plot(idx_sub, y_true_sub, color='black', label='Actual')
    plt.title(f'Prediction Intervals h={h} (Sample Day)')
    plt.ylabel('Energy (Wh)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath(f'fig_uncertainty_h{h}.png'))
    plt.close()

# ==============================================================================
# TASK A & B: REPLICATION & CORRECTION
# ==============================================================================

# Reference Values from Candanedo et al. (2017)
# From Table 4/5/6 (Test Set Results)
PAPER_REF = {
    'GBM': {'RMSE': 66.65, 'R2': 0.57},
    'RF':  {'RMSE': 69.42, 'R2': 0.54},
}

def run_paper_replication(config, pipe_config):
    """
    Executes Task A (Replication) and Task B (Correction) with Automated Lights Comparison.
    h=0, Paper Features.
    Models: Linear, SVR, RF, GBM.
    """
    logger.info("\n" + "="*60)
    logger.info("STARTING REPLICATION (TASK A) & CORRECTION (TASK B) - With Automation")
    logger.info("="*60)

    # Load Data Once
    df_raw = pipeline_core.load_data(config['paths']['data'])
    
    models_to_run = {
        'Linear': config['models'].get('linear', {}),
        'SVR': config['models'].get('svr', {}),
        'RF': config['models'].get('random_forest', {}),
        'GBM': config['models'].get('gradient_boosting', {})
    }
    
    results = []
    
    # Iterate over Include Lights vs No Lights
    for lights_scenario in [True, False]:
        scenario_label = "With Lights" if lights_scenario else "Without Lights"
        logger.info(f"\n--- SCENARIO: {scenario_label} ---")

        # 1. Config & Features
        pipe_config.forecast_horizon_h = 0
        pipe_config.include_lights = lights_scenario

        # Use STRICT paper features (no drops, raw only)
        df_paper = create_strict_paper_features(df_raw, pipe_config)

        logger.info(f"Features: {list(df_paper.columns)}")

        # --- TASK A: Stratified Random Split (Replication) ---
        logger.info(f">>> TASK A: REPLICATION (Stratified Random) [{scenario_label}] <<<")
        X_train, X_test, y_train, y_test, _ = get_data_split(
            df_paper, pipe_config, split_method='stratified_random'
        )

        for m_name, m_params in models_to_run.items():
            # Use Tuning!
            model = tune_model(m_name, X_train, y_train, split_method='stratified_random', random_state=pipe_config.random_seed)
            y_pred = model.predict(X_test)
            metrics = calculate_metrics(y_test, y_pred)

            row = {
                'Task': 'A (Replication)',
                'Scenario': scenario_label,
                'Split': 'Stratified Random',
                'Model': m_name,
                'RMSE': metrics['RMSE'],
                'R2': metrics['R2'],
                'MAE': metrics['MAE']
            }
            if m_name in PAPER_REF and lights_scenario:
                # Reference is typically "With Lights" unless specified
                row['Paper_RMSE'] = PAPER_REF[m_name]['RMSE']
                row['Paper_R2'] = PAPER_REF[m_name]['R2']
            else:
                row['Paper_RMSE'] = np.nan
                row['Paper_R2'] = np.nan
            results.append(row)

        # --- TASK B: Chronological Split (Correction) ---
        logger.info(f">>> TASK B: CORRECTION (Chronological) [{scenario_label}] <<<")
        X_train, X_test, y_train, y_test, _ = get_data_split(
            df_paper, pipe_config, split_method='chronological'
        )

        for m_name, m_params in models_to_run.items():
            # Use Tuning!
            model = tune_model(m_name, X_train, y_train, split_method='chronological', random_state=pipe_config.random_seed)
            y_pred = model.predict(X_test)
            metrics = calculate_metrics(y_test, y_pred)

            row = {
                'Task': 'B (Correction)',
                'Scenario': scenario_label,
                'Split': 'Chronological',
                'Model': m_name,
                'RMSE': metrics['RMSE'],
                'R2': metrics['R2'],
                'MAE': metrics['MAE'],
                'Paper_RMSE': np.nan,
                'Paper_R2': np.nan
            }
            results.append(row)

    df_res = pd.DataFrame(results)
    
    # Save
    save_path = outpath("replication", "replication_correction_results.csv")
    df_res.to_csv(save_path, index=False)
    logger.info(f"Replication results saved to {save_path}")
    
    # Verify Replication (Task A)
    gbm_task_a = df_res[(df_res['Task'] == 'A (Replication)') & (df_res['Model'] == 'GBM')]
    if not gbm_task_a.empty:
        rmse_a = gbm_task_a.iloc[0]['RMSE']
        ref_rmse = PAPER_REF['GBM']['RMSE']
        logger.info(f"Replication GBM RMSE: {rmse_a:.2f} (Paper: {ref_rmse})")

    return df_res


# ==============================================================================
# TASK C: FORECASTING EXTENSION
# ==============================================================================

def run_forecasting_extension(config, pipe_config, h):
    """
    Executes Task C (Forecasting Extension).
    h = 6 (typically).
    Chronological Split.
    Strong Baselines + Modern Model (LGBM).
    Uncertainty + Stability.
    """
    logger.info("\n" + "="*60)
    logger.info(f"STARTING FORECASTING EXTENSION (TASK C) - h={h}")
    logger.info("="*60)
    
    # 1. Load & Engineer Features (with proper horizons)
    pipe_config.forecast_horizon_h = h
    # Enforce current_load_available=False for Task C
    pipe_config.current_load_available = False
    
    df_raw = pipeline_core.load_data(config['paths']['data'])
    df_eng = pipeline_core.create_physics_features(df_raw, pipe_config)
    
    # 2. Chronological Split
    X_train, X_test, y_train, y_test, feature_cols = get_data_split(
        df_eng, pipe_config, split_method='chronological'
    )
    
    # Extract timestamps for baseline lookups
    # Indices are time 't' (issuance).
    # Target time is t + h*freq (assuming 10min freq)
    freq = pd.Timedelta(minutes=10)
    train_indices_target = df_eng.index[:len(y_train)] + (h * freq)
    test_indices_target = df_eng.index[len(y_train):] + (h * freq)
    
    # 3. Baselines
    logger.info("Computing Baselines...")
    
    # Persistence
    y_pred_pers = compute_persistence_baseline(
        y_train, y_test, h=h, current_load_available=pipe_config.current_load_available
    )
    
    # Seasonal Naive
    y_pred_seas_daily = compute_seasonal_naive_baseline(y_train, y_test, 144, h)
    y_pred_seas_weekly = compute_seasonal_naive_baseline(y_train, y_test, 1008, h)
    
    # Time of Week Mean (Target-Aligned)
    y_pred_tow = compute_time_of_week_mean_baseline(
        y_train, y_test, train_indices_target, test_indices_target, h
    )
    
    baselines = {
        'Persistence': y_pred_pers,
        'Seasonal_Daily': y_pred_seas_daily,
        'Seasonal_Weekly': y_pred_seas_weekly,
        'Time_of_Week': y_pred_tow
    }
    
    results = []
    
    # 4. Main Model (LightGBM)
    logger.info("Training LightGBM (Extension)...")
    # Universal Pipeline
    lgbm_params = config['models'].get('lightgbm', {})
    model = create_pipeline('LGBM', model_params=lgbm_params, random_state=pipe_config.random_seed)
    
    # Uncertainty Split (Train -> Calib -> Test)
    # We take last 20% of TRAIN for calibration
    n_train = len(X_train)
    n_calib = int(n_train * 0.2)
    n_prop = n_train - n_calib
    
    X_t_prop = X_train.iloc[:n_prop]
    y_t_prop = y_train[:n_prop]
    X_cal = X_train.iloc[n_prop:]
    y_cal = y_train[n_prop:]
    
    # Fit Conformal
    cp = ConformalPredictor(model, alpha=0.1)
    cp.fit(X_t_prop, y_t_prop, X_cal, y_cal)
    
    # Predict
    y_pred_lgbm, y_lower, y_upper = cp.predict(X_test)
    
    # --- COMMON MASK EVALUATION ---
    # Combine all predictions
    all_preds_df = pd.DataFrame(baselines)
    all_preds_df['LightGBM (Conformal)'] = y_pred_lgbm
    
    # Create valid mask (drop rows where ANY model is NaN)
    # Also ensure y_test is valid
    valid_mask = all_preds_df.notna().all(axis=1) & np.isfinite(y_test)
    
    logger.info(f"Evaluating on common mask. Original: {len(y_test)}, Valid: {valid_mask.sum()}")
    
    y_test_valid = y_test[valid_mask]
    all_preds_valid = all_preds_df[valid_mask]
    y_lower_valid = y_lower[valid_mask]
    y_upper_valid = y_upper[valid_mask]
    
    # Eval Loop on VALID set
    for name in all_preds_valid.columns:
        y_p = all_preds_valid[name].values
        m = calculate_metrics(y_test_valid, y_p)

        # Add Uncertainty Metrics for LGBM
        if name == 'LightGBM (Conformal)':
            unc_m = calculate_uncertainty_metrics(y_test_valid, y_lower_valid, y_upper_valid)
            m.update(unc_m)
            
        row = {'Model': name, **m}
        results.append(row)
    
    # 5. Stability / Importance
    logger.info("Computing Feature Importance (Stability)...")
    imp_df = compute_permutation_importance(
        cp.model, X_test, y_test, feature_cols, n_repeats=10
    )
    imp_save_path = outpath("feature_importance", f"importance_h{h}.csv")
    imp_df.to_csv(imp_save_path, index=False)
    
    # Audit
    audit = audit_negative_controls(imp_df, pipe_config.negative_control_cols)
    if not audit['passed']:
        logger.warning(f"Feature Importance Audit Warnings: {audit['warnings']}")

    # 6. Save Results
    df_res = pd.DataFrame(results)
    save_path = outpath(f"forecasting_results_h{h}.csv")
    df_res.to_csv(save_path, index=False)
    
    # Plotting
    plot_predictions(y_test, y_pred_lgbm, test_indices_target, 'LightGBM', h)
    plot_uncertainty(y_test, y_lower, y_upper, test_indices_target, 0.1, h)
    
    return df_res


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--forecast-horizon', type=int, default=6)
    parser.add_argument('--skip-replication', action='store_true')
    parser.add_argument('--skip-optimization', action='store_true')
    parser.add_argument('--no-current-load', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Run-scoped output setup
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path("outputs") / f"run_{run_id}"
    set_output_dir(str(run_dir))
    ensure_dirs(run_dir)
    
    # 1. Pipeline Config Base
    settings = config.get('settings', {})
    fe = config.get('feature_engineering', {}) or {}

    pipe_config = PipelineConfig(
        target_col=settings['target_col'],
        time_col=settings.get('time_col', 'date'),
        test_size_percent=float(settings.get('test_size_percent', 0.25)),
        random_seed=int(settings.get('random_seed', 42)),
        forecast_horizon_h=args.forecast_horizon,
        include_lights=bool(settings.get('include_lights', True)),
        lags=[int(x) for x in fe.get('lags', [1, 2, 3, 6, 12, 24, 36])],
        rolling_windows=[int(x) for x in fe.get('rolling_windows', [6, 12, 24])],
        current_load_available=(not args.no_current_load) and bool(
            config.get('assumptions', {}).get('current_load_available', True)
        )
    )

    # 2. Run Replication (Task A & B)
    if not args.skip_replication:
        run_paper_replication(config, pipe_config)
        
    # 3. Run Forecasting Extension (Task C)
    run_forecasting_extension(config, pipe_config, h=pipe_config.forecast_horizon_h)
    
    # 4. Optimization (Demo)
    if not args.skip_optimization:
        logger.info("\nRunning Optimization Demo...")
        # Placeholder for demo if needed
        pass

    logger.info("PIPELINE COMPLETE.")
