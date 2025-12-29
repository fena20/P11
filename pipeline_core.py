"""
==============================================================================
PIPELINE CORE: Rigorous Time-Series Forecasting Framework
==============================================================================
Centralizes leakage-free feature engineering, chronological splitting, 
model training, and evaluation logic for h-step-ahead forecasting.

METHODOLOGICAL NOTES:
- Tree-based models (RF/XGB/LGBM) do NOT require feature scaling
- Ridge requires scaling, handled via sklearn Pipeline in nested CV
- Permutation importance uses consistent y-scale (model predicts in same scale)
- Current load y_t IS available at issuance time (see FEATURE_AVAILABILITY.md)

Author: Data Science Team
Date: November 2024 (Revised December 2024)
==============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass, field
import logging
import warnings
from scipy import stats

# Machine Learning
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.inspection import permutation_importance

# Optional dependencies with graceful fallback
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import torch
    HAS_TORCH = True
except Exception:
    # Some environments raise OSError/CUDA-related exceptions, not ImportError.
    HAS_TORCH = False

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class PipelineConfig:
    """
    Configuration for the forecasting pipeline.
    
    CRITICAL: forecast_horizon_h defines the h-step-ahead prediction target.
    All features at time t predict y_{t+h}.
    
    FEATURE AVAILABILITY ASSUMPTION:
    At issuance time t, the following are available:
    - All sensor readings at time t (temperature, humidity, etc.)
    - Current load measurement Appliances(t) if current_load_available=True (real-time metering)
    - Weather observations at time t
    This is consistent with a real-time metering scenario.
    """
    target_col: str = 'Appliances'
    time_col: str = 'date'
    test_size_percent: float = 0.25
    validation_size_percent: float = 0.20
    random_seed: int = 42
    
    # Forecast Horizon
    forecast_horizon_h: int = 6  # Default: 1-hour-ahead for 10-min data

    # Operational assumption
    current_load_available: bool = True  # True if y_t is available at issuance time
    
    # Feature Engineering
    lags: List[int] = field(default_factory=lambda: [1, 2, 3, 6, 12, 24, 36])
    rolling_windows: List[int] = field(default_factory=lambda: [6, 12, 24])
    include_lights: bool = True
    
    # Negative control columns
    negative_control_cols: List[str] = field(default_factory=lambda: ['rv1', 'rv2'])
    
    # CV Settings
    n_outer_folds: int = 5
    n_inner_folds: int = 3
    
    # Model Settings
    use_full_hyperparameter_search: bool = True
    n_iter_random_search: int = 20
    
    def __post_init__(self):
        np.random.seed(self.random_seed)
        if HAS_TORCH:
            torch.manual_seed(self.random_seed)


# ==============================================================================
# DATA LOADING & LEAKAGE-FREE FEATURE ENGINEERING
# ==============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Loads and preprocesses the raw data."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    
    df.sort_index(inplace=True)
    return df


def create_forecast_target(df: pd.DataFrame, target_col: str, h: int) -> pd.DataFrame:
    """
    Creates the h-step-ahead forecast target.
    
    At time t, we predict y_{t+h}.
    """
    df = df.copy()
    df['target_ahead'] = df[target_col].shift(-h)
    logger.info(f"Created {h}-step-ahead target (target_ahead = y_{{t+{h}}})")
    return df


def create_physics_features(df_input: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    Creates physics-informed features with STRICT LEAKAGE PREVENTION.
    
    FEATURE AVAILABILITY AT ISSUANCE TIME t:
    - Current sensor readings: T1-T9, RH_1-RH_9, T_out, etc. (measured at t)
    - Current load: if current_load_available=True, Appliances(t) is available at time t (real-time metering)
    - Weather: T_out(t), RH_out(t), etc. (observed, not forecast)
    
    Note: We use LAGGED target values (y_{t-1}, y_{t-2}, ...) as features.
    The raw Appliances(t) column is dropped from predictors, but its information
    is captured in lag1 which equals y_{t-1}.
    
    For EXOGENOUS VARIABLES: We assume "causal" availability only (observed at t).
    No future weather forecasts are used.
    """
    df = df_input.copy()
    
    # 1. Create h-step-ahead target
    df = create_forecast_target(df, config.target_col, config.forecast_horizon_h)
    
    # 2. Cyclical Time Features
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 3. Thermodynamic Features (current readings at time t)
    indoor_temps = [c for c in df.columns if c.startswith('T') and len(c) == 2 and c[1].isdigit()]
    if indoor_temps:
        df['T_indoor_avg'] = df[indoor_temps].mean(axis=1)
        df['T_indoor_std'] = df[indoor_temps].std(axis=1)
        if 'T_out' in df.columns:
            df['DeltaT'] = df['T_indoor_avg'] - df['T_out']
            df['DeltaT_abs'] = np.abs(df['DeltaT'])
    
    rh_cols = [c for c in df.columns if c.startswith('RH_') and c[3:].isdigit()]
    if rh_cols:
        df['RH_indoor_avg'] = df[rh_cols].mean(axis=1)
    
    # 4. Lag Features on TARGET (Causal)
    # At time t, lag1 = y_{t-1}. This captures "most recent known load".
    target = config.target_col
    # If y_t is available at issuance time, expose it explicitly.
    if config.current_load_available:
        df[f'{target}_lag0'] = df[target]
    for lag in config.lags:
        df[f'{target}_lag{lag}'] = df[target].shift(lag)
    # 5. Rolling Features on TARGET (Causal)
    # If y_t is available, rolling can include time t (shift=0); otherwise use t-1 (shift=1).
    hist_shift = 0 if config.current_load_available else 1
    shifted_target = df[target].shift(hist_shift)
    
    for window in config.rolling_windows:
        df[f'{target}_roll{window}_mean'] = shifted_target.rolling(window=window).mean()
        df[f'{target}_roll{window}_std'] = shifted_target.rolling(window=window).std()
        df[f'{target}_roll{window}_min'] = shifted_target.rolling(window=window).min()
        df[f'{target}_roll{window}_max'] = shifted_target.rolling(window=window).max()

    # 6. Rate of change features (ONLY if current load y_t is available at issuance time)
    if config.current_load_available:
        df[f'{target}_diff1'] = df[target].diff(1)
        df[f'{target}_diff6'] = df[target].diff(6)
    else:
        logger.info("current_load_available=False -> skipping diff features that use y_t")

    
    # 7. Seasonal lags
    df[f'{target}_lag144'] = df[target].shift(144)  # 1 day ago
    df[f'{target}_lag1008'] = df[target].shift(1008)  # 1 week ago
    
    # 8. Handle lights
    if not config.include_lights and 'lights' in df.columns:
        df.drop(columns=['lights'], inplace=True)
        
    # 9. Negative controls
    if 'rv1' in df.columns or 'rv2' in df.columns:
        logger.warning("Random control variables (rv1, rv2) present for diagnostic.")
    
    # Drop NaNs
    original_len = len(df)
    df.dropna(inplace=True)
    logger.info(f"Dropped {original_len - len(df)} rows due to lag/rolling/target generation.")
    
    return df


def check_leakage(df: pd.DataFrame, target_col: str, h: int):
    """Automated leakage check for h-step-ahead forecasting."""
    logger.info("Running automated leakage check...")
    
    if not df.index.is_monotonic_increasing:
        raise ValueError("CRITICAL: Index is not strictly monotonic increasing.")

    if f'{target_col}_lag1' in df.columns:
        idx_loc = len(df) // 2
        if idx_loc > 0:
            val_lag1_t = df.iloc[idx_loc][f'{target_col}_lag1']
            val_target_prev = df.iloc[idx_loc - 1][target_col]
            
            if not np.isclose(val_lag1_t, val_target_prev):
                raise AssertionError("CRITICAL: Lag feature mismatch detected.")

    # If lag0 exists, it must equal y_t at the same row.
    if f'{target_col}_lag0' in df.columns:
        idx_loc = len(df) // 2
        val_lag0_t = df.iloc[idx_loc][f'{target_col}_lag0']
        val_target_t = df.iloc[idx_loc][target_col]
        if not np.isclose(val_lag0_t, val_target_t):
            raise AssertionError('CRITICAL: lag0 feature mismatch detected.')

    if 'target_ahead' in df.columns:
        logger.info(f"target_ahead column present. Forecast horizon h={h}.")

    logger.info("Leakage check passed.")


def audit_negative_controls(feature_importance_df: pd.DataFrame, 
                           negative_control_cols: List[str],
                           threshold_percentile: float = 50) -> Dict[str, Any]:
    """Audits negative control variables in feature importance."""
    audit_results = {
        'passed': True,
        'warnings': [],
        'control_rankings': {}
    }
    
    if feature_importance_df.empty:
        return audit_results
    
    total_features = len(feature_importance_df)
    importance_threshold = np.percentile(feature_importance_df['Importance'], threshold_percentile)
    
    for control_col in negative_control_cols:
        matches = feature_importance_df[feature_importance_df['Feature'].str.contains(control_col, na=False)]
        
        for _, row in matches.iterrows():
            rank = feature_importance_df[feature_importance_df['Importance'] >= row['Importance']].shape[0]
            percentile = (1 - rank / total_features) * 100
            
            audit_results['control_rankings'][row['Feature']] = {
                'importance': row['Importance'],
                'rank': rank,
                'percentile': percentile
            }
            
            if row['Importance'] > importance_threshold:
                audit_results['passed'] = False
                audit_results['warnings'].append(
                    f"WARNING: {row['Feature']} has importance {row['Importance']:.4f} "
                    f"(rank {rank}/{total_features}). Random controls should have near-zero importance!"
                )
    
    return audit_results


# ==============================================================================
# MODEL WRAPPERS
# ==============================================================================

class SafeXGBRegressor(BaseEstimator, RegressorMixin):
    """XGBoost wrapper. Note: Tree models do NOT require feature scaling."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = None
        
    def fit(self, X, y, eval_set=None, verbose=False):
        if HAS_XGB:
            self.model = xgb.XGBRegressor(**self.kwargs)
            self.model.fit(X, y, verbose=verbose)
        else:
            logger.warning("XGBoost not installed. Falling back to RandomForest.")
            self.model = RandomForestRegressor(n_estimators=100, max_depth=10)
            self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        return self.kwargs
    
    def set_params(self, **params):
        self.kwargs.update(params)
        return self
    
    @property
    def feature_importances_(self):
        return self.model.feature_importances_


class SafeLGBMRegressor(BaseEstimator, RegressorMixin):
    """LightGBM wrapper. Note: Tree models do NOT require feature scaling."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = None
        
    def fit(self, X, y, eval_set=None, **fit_params):
        if HAS_LGB:
            self.model = lgb.LGBMRegressor(**self.kwargs)
            self.model.fit(X, y)
        else:
            logger.warning("LightGBM not installed. Falling back to RandomForest.")
            self.model = RandomForestRegressor(n_estimators=100, max_depth=10)
            self.model.fit(X, y)
        return self
        
    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        return self.kwargs
    
    def set_params(self, **params):
        self.kwargs.update(params)
        return self

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


class ConformalPredictor:
    """Split Conformal Prediction for uncertainty quantification."""
    def __init__(self, model, alpha=0.1):
        self.model = model
        self.alpha = alpha
        self.q_hat = None

    def fit(self, X_train, y_train, X_calib, y_calib):
        self.model.fit(X_train, y_train)
        y_pred_calib = self.model.predict(X_calib)
        scores = np.abs(y_calib - y_pred_calib)
        
        n = len(y_calib)
        q_val = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_val = np.clip(q_val, 0, 1)
        self.q_hat = np.quantile(scores, q_val, method='higher')
        
        return self

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred, y_pred - self.q_hat, y_pred + self.q_hat


# ==============================================================================
# BASELINE MODELS
# ==============================================================================

def compute_persistence_baseline(
    y_train: np.ndarray,
    y_test: np.ndarray,
    h: int = 1,
    current_load_available: bool = True
) -> np.ndarray:
    """
    Persistence baseline for an h-step-ahead target stored as target_ahead = y_{t+h}.

    - If current_load_available=True (real-time metering), we assume y_t is known at issuance time t:
        ŷ_{t+h} = y_t
      In target_ahead indexing, this corresponds to shifting back by h rows.

    - If current_load_available=False, we assume the latest available metered load is y_{t-1}:
        ŷ_{t+h} = y_{t-1}
      In target_ahead indexing, this corresponds to shifting back by h+1 rows.

    Note: This baseline must be consistent with the feature availability assumption used in the model.
    """
    n_test = len(y_test)
    y_all = np.concatenate([y_train, y_test])
    test_start = len(y_train)

    lag = 0 if current_load_available else 1
    offset = h + lag

    if test_start - offset < 0:
        raise ValueError("Not enough history to compute persistence baseline with the given horizon/lag.")

    y_pred_persistence = y_all[test_start - offset: test_start - offset + n_test]
    return y_pred_persistence



def compute_seasonal_naive_baseline(y_train: np.ndarray, y_test: np.ndarray,
                                     seasonal_period: int = 144) -> np.ndarray:
    """Seasonal naïve baseline: ŷ_{t+h} = y_{t+h-seasonal_period}"""
    n_test = len(y_test)
    y_all = np.concatenate([y_train, y_test])
    test_start = len(y_train)
    
    y_pred_seasonal = y_all[test_start - seasonal_period: test_start - seasonal_period + n_test]
    
    return y_pred_seasonal


# ==============================================================================
# EVALUATION METRICS
# ==============================================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                      set_name: str = "Test") -> Dict[str, float]:
    """Calculate comprehensive forecasting metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    y_mean = np.mean(y_true)
    
    nrmse = rmse / y_mean if y_mean != 0 else np.inf
    cv_rmse = (rmse / y_mean) * 100 if y_mean != 0 else np.inf
    
    mask_nonzero = y_true != 0
    mape = np.mean(np.abs((y_true[mask_nonzero] - y_pred[mask_nonzero]) / y_true[mask_nonzero])) * 100 if np.any(mask_nonzero) else np.nan
    
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask_denom = denominator != 0
    smape = np.mean(2 * np.abs(y_true[mask_denom] - y_pred[mask_denom]) / denominator[mask_denom]) * 100 if np.any(mask_denom) else np.nan
    
    mbe = np.mean(y_pred - y_true)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'NRMSE': nrmse,
        'CV_RMSE': cv_rmse,
        'MAPE': mape,
        'SMAPE': smape,
        'MBE': mbe
    }


def calculate_uncertainty_metrics(y_true: np.ndarray, y_lower: np.ndarray, 
                                   y_upper: np.ndarray, alpha: float = 0.1) -> Dict[str, float]:
    """Calculate uncertainty metrics for prediction intervals."""
    covered = (y_true >= y_lower) & (y_true <= y_upper)
    picp = np.mean(covered) * 100
    
    width = y_upper - y_lower
    mpiw = np.mean(width)
    
    y_range = np.max(y_true) - np.min(y_true)
    nmpiw = mpiw / y_range if y_range != 0 else np.inf
    
    score = width.copy()
    mask_lower = y_true < y_lower
    score[mask_lower] += (2/alpha) * (y_lower[mask_lower] - y_true[mask_lower])
    mask_upper = y_true > y_upper
    score[mask_upper] += (2/alpha) * (y_true[mask_upper] - y_upper[mask_upper])
    winkler = np.mean(score)
    
    return {'PICP': picp, 'MPIW': mpiw, 'NMPIW': nmpiw, 'Winkler': winkler}


def calculate_error_breakdown(y_true: np.ndarray, y_pred: np.ndarray,
                               timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Calculate error breakdown by temporal categories."""
    residuals = y_true - y_pred
    
    df_analysis = pd.DataFrame({
        'residual': residuals,
        'abs_error': np.abs(residuals),
        'squared_error': residuals ** 2,
        'hour': timestamps.hour,
        'day_of_week': timestamps.dayofweek,
        'month': timestamps.month,
        'is_weekend': timestamps.dayofweek >= 5
    })
    
    results = []
    
    for hour in range(24):
        mask = df_analysis['hour'] == hour
        if mask.sum() > 0:
            subset = df_analysis[mask]
            results.append({
                'Category': 'Hour', 'Value': hour, 'N': mask.sum(),
                'RMSE': np.sqrt(subset['squared_error'].mean()),
                'MAE': subset['abs_error'].mean(),
                'MBE': subset['residual'].mean()
            })
    
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for dow in range(7):
        mask = df_analysis['day_of_week'] == dow
        if mask.sum() > 0:
            subset = df_analysis[mask]
            results.append({
                'Category': 'DayOfWeek', 'Value': day_names[dow], 'N': mask.sum(),
                'RMSE': np.sqrt(subset['squared_error'].mean()),
                'MAE': subset['abs_error'].mean(),
                'MBE': subset['residual'].mean()
            })
    
    for is_wknd, label in [(True, 'Weekend'), (False, 'Weekday')]:
        mask = df_analysis['is_weekend'] == is_wknd
        if mask.sum() > 0:
            subset = df_analysis[mask]
            results.append({
                'Category': 'WeekendFlag', 'Value': label, 'N': mask.sum(),
                'RMSE': np.sqrt(subset['squared_error'].mean()),
                'MAE': subset['abs_error'].mean(),
                'MBE': subset['residual'].mean()
            })
    
    for month in range(1, 13):
        mask = df_analysis['month'] == month
        if mask.sum() > 0:
            subset = df_analysis[mask]
            results.append({
                'Category': 'Month', 'Value': month, 'N': mask.sum(),
                'RMSE': np.sqrt(subset['squared_error'].mean()),
                'MAE': subset['abs_error'].mean(),
                'MBE': subset['residual'].mean()
            })
    
    return pd.DataFrame(results)


def analyze_residual_drift(residuals: np.ndarray, timestamps: pd.DatetimeIndex,
                           window_size: int = 144) -> Dict[str, Any]:
    """Analyze temporal drift patterns in residuals."""
    residual_series = pd.Series(residuals, index=timestamps)
    rolling_bias = residual_series.rolling(window=window_size, min_periods=1).mean()
    
    n = len(residuals)
    q1, q2, q3 = n // 4, n // 2, 3 * n // 4
    
    return {
        'overall_bias': float(np.mean(residuals)),
        'bias_std': float(np.std(residuals)),
        'first_quarter_bias': float(np.mean(residuals[:q1])),
        'second_quarter_bias': float(np.mean(residuals[q1:q2])),
        'third_quarter_bias': float(np.mean(residuals[q2:q3])),
        'fourth_quarter_bias': float(np.mean(residuals[q3:])),
        'max_rolling_bias': float(rolling_bias.max()),
        'min_rolling_bias': float(rolling_bias.min()),
        'bias_drift_range': float(rolling_bias.max() - rolling_bias.min())
    }


def diebold_mariano_test(y_true: np.ndarray, y_pred1: np.ndarray, 
                          y_pred2: np.ndarray, h: int = 1, 
                          criterion: str = "MSE") -> Tuple[float, float]:
    """Diebold-Mariano test for comparing predictive accuracy."""
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2
    
    d = e1**2 - e2**2 if criterion == "MSE" else np.abs(e1) - np.abs(e2)
        
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    
    dm_stat = d_mean / np.sqrt(d_var / len(d))
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return dm_stat, p_value


# ==============================================================================
# NESTED CROSS-VALIDATION (FIXED: Leakage-tight scaling)
# ==============================================================================

def get_hyperparameter_grids() -> Dict[str, Dict]:
    """Returns hyperparameter search spaces."""
    return {
        'Ridge': {'ridge__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
        'RF': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_leaf': [1, 2, 5]
        },
        'XGB': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.05, 0.1]
        },
        'LGBM': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 10, -1],
            'learning_rate': [0.01, 0.05, 0.1]
        }
    }


def create_model_with_pipeline(model_name: str, random_state: int = 42):
    """
    Create model, optionally wrapped in Pipeline for scaling.
    
    CRITICAL FIX: 
    - Tree models (RF, XGB, LGBM) do NOT need scaling
    - Ridge needs scaling, so we wrap in Pipeline
    - This ensures scaling is refit within each inner CV fold
    """
    if model_name == 'Ridge':
        # Ridge needs scaling - wrap in Pipeline for proper nested CV
        return Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge())
        ])
    elif model_name == 'RF':
        return RandomForestRegressor(n_jobs=-1, random_state=random_state)
    elif model_name == 'XGB':
        return SafeXGBRegressor(n_jobs=-1, random_state=random_state, verbosity=0)
    elif model_name == 'LGBM':
        return SafeLGBMRegressor(n_jobs=-1, random_state=random_state, verbose=-1)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_nested_cv(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    Executes TRUE Nested Time-Series Cross-Validation.
    
    METHODOLOGICAL FIXES:
    1. Tree models (RF/XGB/LGBM) are NOT scaled (unnecessary, avoids leakage concern)
    2. Ridge uses Pipeline with scaler, so scaling is refit per inner fold
    3. All models train and predict in ORIGINAL y-scale (no y-scaling for trees)
    """
    logger.info("Starting TRUE Nested Cross-Validation...")
    
    drop_cols = [config.target_col, config.time_col, 'date', 'target_ahead']
    if 'lights' in df.columns and not config.include_lights:
        drop_cols.append('lights')
        
    feature_cols = [c for c in df.columns if c not in drop_cols and c in df.columns]
    
    if 'target_ahead' not in df.columns:
        raise ValueError("target_ahead column not found.")
    
    X = df[feature_cols].values
    y = df['target_ahead'].values  # Original scale
    
    tscv_outer = TimeSeriesSplit(n_splits=config.n_outer_folds)
    tscv_inner = TimeSeriesSplit(n_splits=config.n_inner_folds)
    
    results = []
    model_names = ['Ridge', 'RF', 'XGB', 'LGBM']
    param_grids = get_hyperparameter_grids()
    
    outer_fold = 0
    for train_idx, test_idx in tscv_outer.split(X):
        outer_fold += 1
        logger.info(f"Processing Outer Fold {outer_fold}/{config.n_outer_folds}...")
        
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]
        
        for model_name in model_names:
            logger.info(f"  Training {model_name}...")
            
            # Create model (Pipeline for Ridge, raw for trees)
            base_model = create_model_with_pipeline(model_name, config.random_seed)
            
            if config.use_full_hyperparameter_search and model_name in param_grids:
                param_grid = param_grids[model_name]
                n_combinations = 1
                for v in param_grid.values():
                    n_combinations *= len(v)
                n_iter = min(config.n_iter_random_search, n_combinations)
                
                try:
                    search = RandomizedSearchCV(
                        base_model,
                        param_distributions=param_grid,
                        n_iter=n_iter,
                        cv=tscv_inner,
                        scoring='neg_root_mean_squared_error',
                        random_state=config.random_seed,
                        n_jobs=-1,
                        refit=True
                    )
                    # NO external scaling - Pipeline handles it for Ridge
                    search.fit(X_train_outer, y_train_outer)
                    best_model = search.best_estimator_
                    best_params = search.best_params_
                except Exception as e:
                    logger.warning(f"  Inner CV failed for {model_name}: {e}")
                    best_model = base_model
                    best_model.fit(X_train_outer, y_train_outer)
                    best_params = {}
            else:
                best_model = base_model
                best_model.fit(X_train_outer, y_train_outer)
                best_params = {}
            
            # Predictions in original scale
            y_pred_test = best_model.predict(X_test_outer)
            y_pred_train = best_model.predict(X_train_outer)
            
            metrics_test = calculate_metrics(y_test_outer, y_pred_test)
            metrics_train = calculate_metrics(y_train_outer, y_pred_train)
            
            results.append({
                'Outer_Fold': outer_fold,
                'Model': model_name,
                'Best_Params': str(best_params),
                'Train_RMSE': metrics_train['RMSE'],
                'Train_R2': metrics_train['R2'],
                'Test_RMSE': metrics_test['RMSE'],
                'Test_MAE': metrics_test['MAE'],
                'Test_R2': metrics_test['R2'],
                'Test_NRMSE': metrics_test['NRMSE'],
                'Test_SMAPE': metrics_test['SMAPE'],
                'Test_MBE': metrics_test['MBE'],
                'N_Train': len(y_train_outer),
                'N_Test': len(y_test_outer)
            })
            
    return pd.DataFrame(results)


def get_chronological_split(df: pd.DataFrame, config: PipelineConfig) -> Tuple:
    """Returns (X_train, X_test, y_train, y_test, feature_cols)."""
    drop_cols = [config.target_col, config.time_col, 'date', 'target_ahead']
    if 'lights' in df.columns and not config.include_lights:
        drop_cols.append('lights')
        
    feature_cols = [c for c in df.columns if c not in drop_cols and c in df.columns]
    target_col = 'target_ahead' if 'target_ahead' in df.columns else config.target_col
    
    test_size = int(len(df) * config.test_size_percent)
    train_size = len(df) - test_size
    
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    
    X_train = train[feature_cols].values
    y_train = train[target_col].values
    X_test = test[feature_cols].values
    y_test = test[target_col].values
    
    logger.info(f"Chronological split: Train={train_size}, Test={test_size}")
    
    return X_train, X_test, y_train, y_test, feature_cols


def compute_permutation_importance(model, X_test: Union[np.ndarray, pd.DataFrame], 
                                    y_test: np.ndarray,
                                    feature_cols: List[str], n_repeats: int = 10,
                                    random_state: int = 42) -> pd.DataFrame:
    """
    Compute permutation importance on the TEST set.
    
    Parameters:
    -----------
    model : fitted model
    X_test : array-like or DataFrame with feature names
    y_test : array of true values
    feature_cols : list of feature names
    
    Note on time-series: Naive permutation breaks autocorrelation structure.
    For strict validity, block permutation should be used.
    """
    logger.info("Computing permutation importance on test set...")
    
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )
    
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': result.importances_mean,
        'Importance_Std': result.importances_std
    })
    
    return importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
