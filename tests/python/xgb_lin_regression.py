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
from xgb_synthetic_tabular_data_generator import SyntheticTabularDataDfGenerator


class ModelBuilder:
    def __init__(
        self,
        train_df,
        test_df,
        selected_features,
        target,
        params,
        num_boost_round=200,
        early_stopping_rounds=20,
        n_folds=3,
        random_state=42,
    ):
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
        self.best_features_df = None

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
                evals=[(dv, "val")],
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

        metrics_df = self.create_metrics_df(
            y_test, self.y_pred_test, self.test_base_mean
        )

        performance_df = pd.DataFrame(
            {
                "metric": ["Train R2", "CV Val R2", "Test R2"],
                "value": [self.train_r2, self.cv_val_r2, self.test_r2],
            }
        )

        # Compute feature importance
        importance_gain = self.model.get_score(importance_type="gain")
        if importance_gain:
            total_gain = sum(importance_gain.values())
            normalized_gain = {
                feat: gain / total_gain for feat, gain in importance_gain.items()
            }
            self.best_features_df = pd.DataFrame(
                {
                    "feature": list(normalized_gain.keys()),
                    "importance_gain_normalized": list(normalized_gain.values()),
                }
            ).sort_values(by="importance_gain_normalized", ascending=False)
            self.best_features_df["importance_rank"] = range(
                1, len(self.best_features_df) + 1
            )
            self.best_features_df = self.best_features_df.set_index("importance_rank")
        else:
            self.best_features_df = pd.DataFrame()

        return {
            "model": self.model,
            "metrics_df": metrics_df,
            "performance_df": performance_df,
            "test_base_mean": self.test_base_mean,
            "best_features_df": self.best_features_df,
        }

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
            table_rows.append(
                {
                    "percentile": f"P{p}",
                    "cutoff": round(cutoff, 4),
                    "avg_pred": round(avg_pred, 4),
                    "avg_actual": round(avg_actual, 4),
                    "mae": round(mae_val, 4),
                    "rmse": round(rmse_val, 4),
                    "lift": round(lift, 2),
                }
            )
        metrics_df = pd.DataFrame(table_rows).set_index("percentile")
        return metrics_df


# Create synthetic dataset using generator
generator = SyntheticTabularDataDfGenerator()
tabular_data_df = generator.generate("reg:squarederror")

print("=== tabular_data_df (first 10 rows) ===")
print(tabular_data_df.head(10))

# Hardcoded configurations (update these based on results from the maximizer script)
target = "target"
selected_features = ["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"]
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    # Add/update other params as needed, e.g., 'reg_alpha': 0.001, etc.
}

splitter = TrainTestSplitter(
    tabular_data_df,
    selected_features,
    target="target",
    xgb_objective="reg:squarederror",
)
train_df, test_df = splitter.random_split(test_size=0.2, random_state=42)

builder = ModelBuilder(train_df, test_df, selected_features, target, params)
results = builder.build()
model = results["model"]
metrics_df = results["metrics_df"]
performance_df = results["performance_df"]
test_base_mean = results["test_base_mean"]
best_features_df = results["best_features_df"]

# Modify for printing
performance_df["metric"] = performance_df["metric"] + ":"

# Outputs
print("\n=== Model Performance ===")
print(performance_df.to_string(index=False))

print(f"\nBase mean on test set: {test_base_mean:.4f}")

print("\n=== best_features_df (Top Features by Gain) ===")
print(best_features_df.to_string(float_format="{:.4f}".format))

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
print(
    "- 'Pxx' means selecting samples with predicted value >= xx-th percentile (i.e., top (100-xx)%)"
)
print("- Higher percentile = stricter threshold = higher lift, lower count")
print("- Lift = avg_actual / base_mean")
