# ~/Apps/rtutor/tests/python/xgb_lin_regression.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold
from xgb_train_test_splitter import TrainTestSplitter

# Set seed for reproducibility
np.random.seed(42)

# Create synthetic dataset
n_samples = 10000
n_features = 20

features = [f"feat_{i}" for i in range(n_features)]
X = pd.DataFrame(
    np.random.normal(0, 1, size=(n_samples, n_features)),
    columns=features,
    index=pd.RangeIndex(n_samples, name="id"),
)

# True underlying relationship
true_intercept = 1.0
true_coeffs = np.zeros(n_features)
true_coeffs[0] = 0.5    # Strong positive
true_coeffs[1] = -0.3   # Strong negative
true_coeffs[2] = 0.2    # Moderate positive
true_coeffs[3] = 0.15   # Smaller positive
# Rest are irrelevant (true coeff = 0)

noise = np.random.normal(0, 0.2, n_samples)
linear = true_intercept + X @ true_coeffs + noise
y = pd.Series(np.exp(linear), name="target")

# Combine into one DataFrame
tabular_data_df = X.copy()
tabular_data_df["target"] = y
tabular_data_df['timestamp'] = pd.date_range(start='2023-01-01', periods=n_samples, freq='min')

print("=== tabular_data_df (first 10 rows) ===")
print(tabular_data_df.head(10))

class ModelBuilder:
    def __init__(self, train_df, test_df, selected_features, target, params, num_boost_round=200, early_stopping_rounds=20, n_folds=3, random_state=42):
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        self.selected_features = selected_features
        self.target = target
        self.params = params
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.n_folds = n_folds
        self.random_state = random_state
        self.model = None
        self.train_r2 = None
        self.cv_val_r2 = None
        self.test_r2 = None
        self.test_base_mean = None
        self.y_pred_test = None

    def _compute_cv_r2_best_iter(self, X, y):
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        r2s = []
        best_iters = []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_v = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_v = y.iloc[train_idx], y.iloc[val_idx]
            dtr = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
            dv = xgb.DMatrix(X_v, label=y_v, enable_categorical=True)
            model = xgb.train(
                self.params,
                dtr,
                num_boost_round=self.num_boost_round,
                evals=[(dv, 'val')],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=False,
            )
            pred_v = model.predict(dv)
            r2 = r2_score(y_v, pred_v)
            r2s.append(r2)
            best_iters.append(model.best_iteration)
        return np.mean(r2s), int(np.mean(best_iters))

    def build(self):
        X_train = self.train_df[self.selected_features]
        y_train = self.train_df[self.target]
        X_test = self.test_df[self.selected_features]
        y_test = self.test_df[self.target]

        self.cv_val_r2, best_iter = self._compute_cv_r2_best_iter(X_train, y_train)

        dtrain_full = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)

        self.model = xgb.train(
            self.params,
            dtrain_full,
            num_boost_round=best_iter + 1,
            verbose_eval=False,
        )

        y_pred_train = self.model.predict(dtrain_full)
        self.train_r2 = r2_score(y_train, y_pred_train)

        dtest = xgb.DMatrix(X_test)
        self.y_pred_test = self.model.predict(dtest)
        self.test_r2 = r2_score(y_test, self.y_pred_test)

        self.test_base_mean = y_test.mean()

        metrics_df = self.create_metrics_df(y_test, self.y_pred_test, self.test_base_mean)

        performance_df = pd.DataFrame({
            'metric': ['Train R2', 'CV Val R2', 'Test R2'],
            'value': [self.train_r2, self.cv_val_r2, self.test_r2]
        })

        return {'model': self.model, 'metrics_df': metrics_df, 'performance_df': performance_df, 'test_base_mean': self.test_base_mean}

    def create_metrics_df(self, y_test, y_pred, test_base_mean):
        percentiles = [100, 99] + list(range(95, 0, -5)) + [1, 0]
        table_rows = []
        for p in percentiles:
            cutoff = np.percentile(y_pred, p)
            mask = y_pred >= cutoff
            if not np.any(mask):
                continue
            sub_pred = y_pred[mask]
            sub_actual = y_test[mask]
            avg_pred = np.mean(sub_pred)
            avg_actual = np.mean(sub_actual)
            mae_val = mean_absolute_error(sub_actual, sub_pred)
            rmse_val = root_mean_squared_error(sub_actual, sub_pred)
            lift = avg_actual / test_base_mean if test_base_mean > 0 else 0
            table_rows.append({
                'percentile': f'P{p}',
                'cutoff': round(cutoff, 4),
                'avg_pred': round(avg_pred, 4),
                'avg_actual': round(avg_actual, 4),
                'mae': round(mae_val, 4),
                'rmse': round(rmse_val, 4),
                'lift': round(lift, 2),
            })
        metrics_df = pd.DataFrame(table_rows).set_index('percentile')
        return metrics_df

# Hardcoded configurations (update these based on results from the maximizer script)
target = 'target'
selected_features = features  # Specify your selected features here, e.g., ['feat_0', 'feat_1', ...]
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    # Add/update other params as needed, e.g., 'reg_alpha': 0.001, etc.
}

splitter = TrainTestSplitter(
    tabular_data_df, 
    features, 
    target='target',
    xgb_objective='reg:squarederror'
)
train_df, test_df = splitter.random_split(test_size=0.2, random_state=42)

builder = ModelBuilder(train_df, test_df, selected_features, target, params)
results = builder.build()
model = results['model']
metrics_df = results['metrics_df']
performance_df = results['performance_df']
test_base_mean = results['test_base_mean']

# Modify for printing
performance_df['metric'] = performance_df['metric'] + ':'

# Outputs
print("\n=== Model Performance ===")
print(performance_df.to_string(index=False))

print(f"\nBase mean on test set: {test_base_mean:.4f}")

print("\n=== Performance by Percentile Threshold ===")
print(metrics_df.to_string())

print("\n=== Model Parameters ===")
print(params)

# Demonstrate making a prediction
example_row = tabular_data_df.iloc[0][selected_features]
dexample = xgb.DMatrix(pd.DataFrame([example_row]))
example_pred = model.predict(dexample)[0]

print(f"\n=== Example Prediction ===")
print(f"Input features:\n{selected_features}")
print(f"Predicted target: {example_pred:.4f}")

print("\nNOTE:")
print("- 'Pxx' means selecting samples with predicted value >= xx-th percentile (i.e., top (100-xx)%)")
print("- Higher percentile = stricter threshold = higher lift, lower count")
print("- Lift = avg_actual / base_mean")
