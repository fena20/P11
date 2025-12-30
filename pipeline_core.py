"""
==============================================================================
PIPELINE CORE: Rigorous Time-Series Forecasting Framework
==============================================================================
Centralizes leakage-free feature engineering, chronological splitting, 
model training, and evaluation logic for h-step-ahead forecasting.

METHODOLOGICAL NOTES:
- Universal Preprocessing: All models use ColumnTransformer (StandardScaler + OneHotEncoder).
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, train_test_split, KFold
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
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
    """Creates the h-step-ahead forecast target."""
    df = df.copy()
    df['target_ahead'] = df[target_col].shift(-h)
    logger.info(f"Created {h}-step-ahead target (target_ahead = y_{{t+{h}}})")
    return df


def create_physics_features(df_input: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    Creates physics-informed features with STRICT LEAKAGE PREVENTION.
    """
    df = df_input.copy()
    
    # 1. Create h-step-ahead target
    df = create_forecast_target(df, config.target_col, config.forecast_horizon_h)
    
    # 2. Cyclical Time Features
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute # Added minute for ToW lookup
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # NSM-based time-of-day encoding
    nsm_sec = df.index.hour * 3600 + df.index.minute * 60 + df.index.second
    df['tod_sin'] = np.sin(2 * np.pi * nsm_sec / 86400)
    df['tod_cos'] = np.cos(2 * np.pi * nsm_sec / 86400)

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
    target = config.target_col
    if config.current_load_available:
        df[f'{target}_lag0'] = df[target]
    for lag in config.lags:
        df[f'{target}_lag{lag}'] = df[target].shift(lag)

    # 5. Rolling Features on TARGET (Causal)
    hist_shift = 0 if config.current_load_available else 1
    shifted_target = df[target].shift(hist_shift)
    
    for window in config.rolling_windows:
        df[f'{target}_roll{window}_mean'] = shifted_target.rolling(window=window).mean()
        df[f'{target}_roll{window}_std'] = shifted_target.rolling(window=window).std()
        df[f'{target}_roll{window}_min'] = shifted_target.rolling(window=window).min()
        df[f'{target}_roll{window}_max'] = shifted_target.rolling(window=window).max()

    # 6. Rate of change features
    if config.current_load_available:
        df[f'{target}_diff1'] = df[target].diff(1)
        df[f'{target}_diff6'] = df[target].diff(6)
    else:
        logger.info("current_load_available=False -> skipping diff features that use y_t")

    # 7. Seasonal lags (Strictly causal, these are fine if shifted)
    # Actually, for h>1, we need to be careful.
    # But usually lag144 means y_{t-144}.
    # We will use these for baselines mostly.
    
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

    if f'{target_col}_lag0' in df.columns:
        idx_loc = len(df) // 2
        val_lag0_t = df.iloc[idx_loc][f'{target_col}_lag0']
        val_target_t = df.iloc[idx_loc][target_col]
        if not np.isclose(val_lag0_t, val_target_t):
            raise AssertionError('CRITICAL: lag0 feature mismatch detected.')

    logger.info("Leakage check passed.")


def audit_negative_controls(feature_importance_df: pd.DataFrame, 
                           negative_control_cols: List[str]) -> Dict[str, Any]:
    """
    Audits negative control variables using Z-score stability check.
    Fails only if |Z| > 2 (stable signal).
    """
    audit_results = {
        'passed': True,
        'warnings': [],
        'control_stats': {}
    }
    
    if feature_importance_df.empty:
        return audit_results
    
    # Expect feature_importance_df to have 'Importance' (mean) and 'Importance_Std'
    # if it comes from our robust permutation function.
    # If not, we fall back to raw importance.
    
    for control_col in negative_control_cols:
        matches = feature_importance_df[feature_importance_df['Feature'].str.contains(control_col, na=False)]
        
        for _, row in matches.iterrows():
            mean_imp = row['Importance']
            std_imp = row.get('Importance_Std', 0.0)

            if std_imp > 1e-9:
                z_score = mean_imp / std_imp
            else:
                # If std is ~0, check if mean is significant
                # If mean is large (e.g. > 1e-5), it's a stable signal -> Z = inf (Fail)
                # If mean is tiny, it's stable zero -> Z = 0 (Pass)
                if abs(mean_imp) > 1e-5:
                    z_score = 999.0 # Effectively infinite
                else:
                    z_score = 0.0
            
            audit_results['control_stats'][row['Feature']] = {
                'mean_importance': mean_imp,
                'std_importance': std_imp,
                'z_score': z_score
            }
            
            # Fail if robust signal detected (> 2 sigma)
            if abs(z_score) > 2.0:
                audit_results['passed'] = False
                audit_results['warnings'].append(
                    f"WARNING: {row['Feature']} has stable importance (Z={z_score:.2f}). "
                    f"Mean={mean_imp:.4f}, Std={std_imp:.4f}. Potential overfitting to noise."
                )
    
    return audit_results


# ==============================================================================
# MODEL FACTORY & PIPELINES
# ==============================================================================

class SafeXGBRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = None
    def fit(self, X, y, eval_set=None, verbose=False):
        if HAS_XGB:
            self.model = xgb.XGBRegressor(**self.kwargs)
            self.model.fit(X, y, verbose=verbose)
        else:
            self.model = RandomForestRegressor(n_estimators=100)
            self.model.fit(X, y)
        self.is_fitted_ = True
        return self
    def predict(self, X): return self.model.predict(X)
    def get_params(self, deep=True): return self.kwargs
    def set_params(self, **params):
        self.kwargs.update(params)
        return self
    @property
    def feature_importances_(self): return self.model.feature_importances_

class SafeLGBMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = None
    def fit(self, X, y, eval_set=None, **fit_params):
        if HAS_LGB:
            self.model = lgb.LGBMRegressor(**self.kwargs)
            self.model.fit(X, y)
        else:
            self.model = RandomForestRegressor(n_estimators=100)
            self.model.fit(X, y)
        self.is_fitted_ = True
        return self
    def predict(self, X): return self.model.predict(X)
    def get_params(self, deep=True): return self.kwargs
    def set_params(self, **params):
        self.kwargs.update(params)
        return self
    @property
    def feature_importances_(self): return self.model.feature_importances_

def create_pipeline(model_type: str, model_params: Optional[Dict[str, Any]] = None, random_state: int = 42) -> Pipeline:
    """
    Creates a universal pipeline with robust preprocessing.
    ColumnTransformer ensures correct handling of numeric vs categorical features.

    Policy:
    - Numeric: StandardScaler
    - Categorical: OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    - Applied to ALL models (Linear, SVR, RF, GBM) for consistency.
    """
    if model_params is None:
        model_params = {}

    # Selector for columns
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, make_column_selector(dtype_include=np.number)),
            ('cat', categorical_transformer, make_column_selector(dtype_include=[object, 'category']))
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    # Clean params (remove things that might conflict if needed, or just pass them)
    # Ensure random_state is respected if model accepts it

    if model_type == 'Linear':
        model = LinearRegression(**model_params)
    elif model_type == 'Ridge':
        model = Ridge(random_state=random_state, **model_params)
    elif model_type == 'SVR':
        model = SVR(**model_params)
    elif model_type == 'RF':
        model_params.setdefault('n_jobs', -1)
        if 'random_state' in model_params:
             _ = model_params.pop('random_state')
        model = RandomForestRegressor(random_state=random_state, **model_params)
    elif model_type == 'GBM':
        # Vanilla Gradient Boosting (sklearn)
        if 'random_state' in model_params:
             _ = model_params.pop('random_state')
        model = GradientBoostingRegressor(random_state=random_state, **model_params)
    elif model_type == 'XGB':
        model_params.setdefault('n_jobs', -1)
        model_params.setdefault('verbosity', 0)
        if 'random_state' in model_params:
             _ = model_params.pop('random_state')
        model = SafeXGBRegressor(random_state=random_state, **model_params)
    elif model_type == 'LGBM':
        model_params.setdefault('n_jobs', -1)
        model_params.setdefault('verbose', -1)
        if 'random_state' in model_params:
             _ = model_params.pop('random_state')
        model = SafeLGBMRegressor(random_state=random_state, **model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])


class ConformalPredictor:
    """Split Conformal Prediction with strictly time-ordered calibration."""
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
    Persistence:
    if current_load_available: ŷ_{t+h} = y_t
    else: ŷ_{t+h} = y_{t-1}
    """
    n_test = len(y_test)
    y_all = np.concatenate([y_train, y_test])
    test_start = len(y_train)

    lag = 0 if current_load_available else 1
    offset = h + lag

    if test_start - offset < 0:
        # Not enough history
        return np.full(n_test, np.nan)

    y_pred_persistence = y_all[test_start - offset: test_start - offset + n_test]
    return y_pred_persistence


def compute_seasonal_naive_baseline(y_train: np.ndarray, y_test: np.ndarray,
                                     seasonal_period: int = 144, h: int = 1) -> np.ndarray:
    """
    Strict Seasonal Naive: ŷ_{t+h} = y_{t+h-seasonal_period}
    This means we look back exactly 'seasonal_period' steps from the TARGET time.
    """
    n_test = len(y_test)
    y_all = np.concatenate([y_train, y_test])
    test_start = len(y_train)
    
    # We want y at index (t+h) - period
    # In y_all, index i corresponds to time t+i relative to start of y_all
    # We are predicting for i in [test_start, test_start+n_test-1]
    # Prediction at i uses value at i - seasonal_period

    start_idx = test_start - seasonal_period
    end_idx = test_start + n_test - seasonal_period
    
    if start_idx < 0:
        # Prepend NaNs if history missing
        missing_count = -start_idx
        available = y_all[0:end_idx]
        return np.concatenate([np.full(missing_count, np.nan), available])

    return y_all[start_idx:end_idx]


def compute_time_of_week_mean_baseline(y_train: np.ndarray, y_test: np.ndarray,
                                       timestamps_train: pd.DatetimeIndex,
                                       timestamps_test: pd.DatetimeIndex,
                                       h: int = 1) -> np.ndarray:
    """
    Time-of-Week Mean Baseline.
    Computes mean load for each (DayOfWeek, Hour, Minute) bucket in TRAIN.
    Predicts using the bucket of the TARGET time (t+h).
    Timestamps passed MUST be the target timestamps.
    """
    # 1. Create Train Lookup
    train_df = pd.DataFrame({'y': y_train})
    train_df['dow'] = timestamps_train.dayofweek
    train_df['hour'] = timestamps_train.hour
    train_df['minute'] = timestamps_train.minute

    lookup = train_df.groupby(['dow', 'hour', 'minute'])['y'].mean()

    # 2. Map Test
    test_df = pd.DataFrame({'dow': timestamps_test.dayofweek,
                            'hour': timestamps_test.hour,
                            'minute': timestamps_test.minute})

    y_pred = test_df.apply(
        lambda row: lookup.get((row['dow'], row['hour'], row['minute']), np.nan), axis=1
    ).values

    # Strict Policy: No Fallback
    # If bucket is missing, return NaN.
    # Evaluation logic handles dropping NaNs consistently.

    return y_pred


# ==============================================================================
# EVALUATION METRICS
# ==============================================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive forecasting metrics."""
    # Strict NaN handling: if any NaN in pred/true, drop
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.all():
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {k: np.nan for k in ['RMSE', 'MAE', 'R2', 'NRMSE', 'CV_RMSE', 'MAPE', 'SMAPE', 'MBE']}

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    y_mean = np.mean(y_true)
    nrmse = rmse / y_mean if y_mean != 0 else np.inf
    cv_rmse = (rmse / y_mean) * 100 if y_mean != 0 else np.inf
    
    mbe = np.mean(y_pred - y_true)
    
    # SMAPE
    denom = np.abs(y_true) + np.abs(y_pred)
    mask_denom = denom != 0
    smape = np.mean(2 * np.abs(y_true[mask_denom] - y_pred[mask_denom]) / denom[mask_denom]) * 100 if np.any(mask_denom) else np.nan

    return {
        'RMSE': rmse, 'MAE': mae, 'R2': r2, 'NRMSE': nrmse,
        'CV_RMSE': cv_rmse, 'SMAPE': smape, 'MBE': mbe
    }


def calculate_uncertainty_metrics(y_true: np.ndarray, y_lower: np.ndarray, 
                                   y_upper: np.ndarray, alpha: float = 0.1) -> Dict[str, float]:
    # Filter NaNs
    mask = np.isfinite(y_true) & np.isfinite(y_lower) & np.isfinite(y_upper)
    y_true, y_lower, y_upper = y_true[mask], y_lower[mask], y_upper[mask]

    if len(y_true) == 0:
        return {'PICP': np.nan, 'MPIW': np.nan}

    covered = (y_true >= y_lower) & (y_true <= y_upper)
    picp = np.mean(covered) * 100
    width = y_upper - y_lower
    mpiw = np.mean(width)
    
    return {'PICP': picp, 'MPIW': mpiw}


def calculate_error_breakdown(y_true: np.ndarray, y_pred: np.ndarray,
                               timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    # Filter NaNs
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred, timestamps = y_true[mask], y_pred[mask], timestamps[mask]

    residuals = y_true - y_pred
    df_analysis = pd.DataFrame({
        'residual': residuals,
        'squared_error': residuals ** 2,
        'abs_error': np.abs(residuals),
        'hour': timestamps.hour,
        'day_of_week': timestamps.dayofweek,
        'is_weekend': timestamps.dayofweek >= 5
    })
    
    results = []
    # Hour breakdown
    for h in range(24):
        sub = df_analysis[df_analysis['hour'] == h]
        if not sub.empty:
            results.append({'Category': 'Hour', 'Value': h,
                            'RMSE': np.sqrt(sub['squared_error'].mean()),
                            'MAE': sub['abs_error'].mean(),
                            'MBE': sub['residual'].mean()})
    return pd.DataFrame(results)


def analyze_residual_drift(residuals: np.ndarray, timestamps: pd.DatetimeIndex,
                           window_size: int = 144) -> Dict[str, Any]:
    # Filter NaNs
    mask = np.isfinite(residuals)
    residuals, timestamps = residuals[mask], timestamps[mask]

    residual_series = pd.Series(residuals, index=timestamps)
    rolling_bias = residual_series.rolling(window=window_size, min_periods=1).mean()
    
    return {
        'overall_bias': float(np.mean(residuals)),
        'bias_std': float(np.std(residuals)),
        'max_rolling_bias': float(rolling_bias.max()),
        'min_rolling_bias': float(rolling_bias.min())
    }

def diebold_mariano_test(y_true, y_pred1, y_pred2):
    # Filter NaNs
    mask = np.isfinite(y_true) & np.isfinite(y_pred1) & np.isfinite(y_pred2)
    y_true, y_pred1, y_pred2 = y_true[mask], y_pred1[mask], y_pred2[mask]

    if len(y_true) == 0: return np.nan, np.nan

    e1 = y_true - y_pred1
    e2 = y_true - y_pred2
    d = e1**2 - e2**2
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    if d_var == 0: return 0.0, 1.0
    dm_stat = d_mean / np.sqrt(d_var / len(d))
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value


# ==============================================================================
# DATA SPLITTING & PAPER MODE
# ==============================================================================

def get_paper_mode_features(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """
    Returns strict Paper Mode feature set.
    Assertion: h=0 (Paper mode is estimation/nowcasting).
    Assertion: No leakage, no derived lags.
    """
    if h != 0:
        raise ValueError(f"CRITICAL: Paper Mode requires h=0 (nowcasting). Got h={h}.")
        
    whitelist = [
        'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'RH_5',
        'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9',
        'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint',
        'lights', 'nsm', 'WeekStatus', 'Day_of_week'
    ]
    
    # We must construct these from raw if not present or ensure they are present
    # Assuming df coming in has raw features + some derived.
    # We select ONLY whitelist columns that exist.
    
    # Re-construct categorical Time features if missing (safeguard)
    # The dataframe index is datetime
    if 'WeekStatus' not in df.columns:
        df['WeekStatus'] = df.index.dayofweek.map(lambda x: 'Weekend' if x >= 5 else 'Weekday')
    if 'Day_of_week' not in df.columns:
        df['Day_of_week'] = df.index.day_name()
    if 'nsm' not in df.columns:
        df['nsm'] = df.index.hour * 3600 + df.index.minute * 60 + df.index.second

    # Select only whitelisted columns
    cols_to_keep = [c for c in whitelist if c in df.columns]
    
    # CHECK FOR LEAKAGE
    for c in cols_to_keep:
        if 'Appliances' in c or 'lag' in c or 'roll' in c or 'diff' in c:
            raise ValueError(f"CRITICAL LEAKAGE: Feature {c} passed whitelist check improperly.")

    # Also verify NO rv1/rv2
    if any(c in cols_to_keep for c in ['rv1', 'rv2']):
        raise ValueError("CRITICAL: Negative controls found in Paper Mode features.")
        
    # Return DF with target_ahead (which is y_t for h=0)
    # But get_data_split expects target_ahead.
    # For h=0, target_ahead = Appliances.

    df_out = df[cols_to_keep].copy()
    if 'target_ahead' in df.columns:
        df_out['target_ahead'] = df['target_ahead']
    elif 'Appliances' in df.columns and h == 0:
        df_out['target_ahead'] = df['Appliances']
        
    return df_out


def get_data_split(df: pd.DataFrame, config: PipelineConfig,
                   split_method: str = 'chronological') -> Tuple:
    """
    Returns (X_train, X_test, y_train, y_test, feature_cols).
    """
    drop_cols = [config.target_col, config.time_col, 'date', 'target_ahead']
    if 'lights' in df.columns and not config.include_lights:
        drop_cols.append('lights')
        
    feature_cols = [c for c in df.columns if c not in drop_cols and c in df.columns]
    target_col = 'target_ahead'
    
    X = df[feature_cols] # Keep as DataFrame for Pipeline (ColumnTransformer)
    y = df[target_col].values

    if split_method == 'stratified_random':
        # Stratified Random Split (75/25) matching CARET
        # Use qcut to bin target
        try:
            bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
        except ValueError:
            # Fallback if too many duplicates
            bins = pd.qcut(y, q=5, labels=False, duplicates='drop')

        logger.info(f"Stratified Random Split: {len(np.unique(bins))} bins used.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size_percent,
            stratify=bins, random_state=config.random_seed, shuffle=True
        )

    elif split_method == 'chronological':
        test_size = int(len(df) * config.test_size_percent)
        train_size = len(df) - test_size
        X_train = X.iloc[:train_size]
        X_test = X.iloc[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]

    else:
        raise ValueError(f"Unknown split method: {split_method}")

    return X_train, X_test, y_train, y_test, feature_cols

def get_chronological_split(df, config):
    return get_data_split(df, config, split_method='chronological')

def compute_permutation_importance(model, X_test, y_test, feature_cols, n_repeats=10, random_state=42):
    # Use sklearn's permutation_importance which handles Pipelines correctly
    r = permutation_importance(model, X_test, y_test, n_repeats=n_repeats,
                               random_state=random_state, n_jobs=-1, scoring='neg_mean_squared_error')

    return pd.DataFrame({
        'Feature': feature_cols,
        'Importance': r.importances_mean,
        'Importance_Std': r.importances_std
    }).sort_values('Importance', ascending=False)

def create_strict_paper_features(df_input: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    Creates feature set for Strict Replication (h=0) WITHOUT dropna/lags.
    Ensures full dataset usage matching the reference paper.
    """
    df = df_input.copy()

    # 1. Target (h=0 means current Appliances)
    df['target_ahead'] = df[config.target_col]
    
    # 2. Time Features (needed for paper)
    # The paper uses: NSM, WeekStatus, Day_of_week
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df['nsm'] = df.index.hour * 3600 + df.index.minute * 60 + df.index.second
    df['WeekStatus'] = df.index.dayofweek.map(lambda x: 'Weekend' if x >= 5 else 'Weekday')
    df['Day_of_week'] = df.index.day_name()

    # 3. Whitelist
    whitelist = [
        'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'RH_5',
        'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9',
        'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint',
        'nsm', 'WeekStatus', 'Day_of_week', 'target_ahead'
    ]

    if config.include_lights:
        whitelist.append('lights')

    # Select columns
    # Use intersection to be safe but check for critical misses
    existing = [c for c in whitelist if c in df.columns]
    
    return df[existing].copy()

def tune_model(model_name: str, X_train, y_train, split_method: str, random_state: int = 42) -> Any:
    """
    Performs inner CV tuning to find best parameters.
    Returns the fitted estimator.
    """
    # Define Parameter Grids (Minimal but representative)
    param_grids = {
        'Linear': {'regressor__fit_intercept': [True, False]},
        'Ridge': {'regressor__alpha': [0.1, 1.0, 10.0]},
        'SVR': {'regressor__C': [0.1, 1.0, 10.0], 'regressor__gamma': ['scale', 'auto']},
        'RF': {'regressor__n_estimators': [50, 100], 'regressor__max_depth': [10, 20, None]},
        'GBM': {'regressor__n_estimators': [50, 100], 'regressor__learning_rate': [0.05, 0.1], 'regressor__max_depth': [3, 5]},
        'LGBM': {'regressor__n_estimators': [50, 100], 'regressor__learning_rate': [0.05, 0.1]}
    }
    
    # 1. Create Base Pipeline
    base_model = create_pipeline(model_name, random_state=random_state)

    # 2. Define CV Strategy
    if split_method == 'stratified_random':
        # KFold for Paper-Style
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    else:
        # TimeSeriesSplit for Chronological
        cv = TimeSeriesSplit(n_splits=5)

    grid = param_grids.get(model_name, {})
    if not grid:
        logger.info(f"No param grid for {model_name}, using default.")
        base_model.fit(X_train, y_train)
        return base_model

    # 3. Tuning
    # Use RandomizedSearchCV for efficiency
    search = RandomizedSearchCV(
        base_model,
        param_distributions=grid,
        n_iter=5, # Keep it fast for this demo
        cv=cv,
        scoring='neg_root_mean_squared_error',
        random_state=random_state,
        n_jobs=-1
    )
    
    try:
        search.fit(X_train, y_train)
        logger.info(f"Tuning {model_name}: Best Params {search.best_params_}")
        return search.best_estimator_
    except Exception as e:
        logger.warning(f"Tuning failed for {model_name}: {e}. Falling back to default.")
        base_model.fit(X_train, y_train)
        return base_model
