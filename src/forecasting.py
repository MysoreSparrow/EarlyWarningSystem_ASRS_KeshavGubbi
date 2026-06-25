# Layer 3 — LightGBM time series forecasting
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from src.logger import get_logger

logger = get_logger(__name__)


def create_ts_features(monthly_df: pd.DataFrame, target_col: str,
                        lags: list = [1, 2, 3, 6, 12],
                        windows: list = [3, 6, 12]) -> pd.DataFrame:
    df = monthly_df.copy().sort_index()
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    for w in windows:
        df[f'rolling_mean_{w}'] = df[target_col].shift(1).rolling(w).mean()
        df[f'rolling_std_{w}'] = df[target_col].shift(1).rolling(w).std()

    df['month'] = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['quarter'] = df.index.quarter

    dates = pd.to_datetime(df.index.to_timestamp() if hasattr(df.index, 'to_timestamp') else df.index)
    df['regime_disruption'] = ((dates >= '2020-03-01') & (dates < '2022-01-01')).astype(int)
    df['regime_recovery'] = ((dates >= '2022-01-01') & (dates < '2024-01-01')).astype(int)

    return df.dropna()


def train_ts_forecast(asrs: pd.DataFrame, category: str,
                       forecast_horizon: int = 6) -> dict:
    mask = asrs['Events | Anomaly'].fillna('').str.contains(category, case=False, na=False)
    monthly = (asrs[mask]
               .groupby(asrs[mask]['date'].dt.to_period('M'))['ACN'].count()
               .to_timestamp().resample('MS').sum())

    monthly_df = pd.DataFrame({'count': monthly})
    monthly_df = create_ts_features(monthly_df, 'count')
    feature_cols = [c for c in monthly_df.columns if c != 'count']

    train = monthly_df[monthly_df.index.year < 2024]
    test = monthly_df[monthly_df.index.year >= 2024]

    model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15,
                               min_data_in_leaf=5, random_state=42, verbose=-1)
    model.fit(train[feature_cols], train['count'])

    test_pred = model.predict(test[feature_cols]) if len(test) > 0 else np.array([])
    mae = mean_absolute_error(test['count'], test_pred) if len(test) > 0 else float('nan')
    logger.info("%s: MAE = %.2f", category, mae)

    return {
        'category': category, 'model': model, 'monthly': monthly,
        'test_actual': test['count'] if len(test) > 0 else pd.Series(dtype=float),
        'test_pred': test_pred, 'mae': mae, 'feature_cols': feature_cols,
        'feature_importance': dict(zip(feature_cols, model.feature_importances_))
    }
