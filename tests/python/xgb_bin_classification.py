# ~/Apps/rtutor/tests/python/xgb_bin_classification.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
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
        self.train_df = train_df
        self.test_df = test_df
        self.selected_features = selected_features
        self.target = target
        self.params = params
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.n_folds = n_folds
        self.random_state = random_state
        self.model = None
        self.train_auc = None
        self.cv_val_auc = None
        self.test_auc = None
        self.test_base_rate = None
        self.y_pred_test = None
        self.best_features_df = None

    def build(self):
        X_train = self.train_df[self.selected_features]
        y_train = self.train_df[self.target]
        X_test = self.test_df[self.selected_features]
        y_test = self.test_df[self.target]

        dtrain_full = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)

        cv_results = xgb.cv(
            self.params,
            dtrain_full,
            num_boost_round=self.num_boost_round,
            nfold=self.n_folds,
            stratified=True,
            early_stopping_rounds=self.early_stopping_rounds,
            metrics="auc",
            seed=self.random_state,
            verbose_eval=False,
        )
        best_iter = cv_results["test-auc-mean"].argmax()
        self.cv_val_auc = cv_results["test-auc-mean"].max()

        self.model = xgb.train(
            self.params,
            dtrain_full,
            num_boost_round=best_iter + 1,
            verbose_eval=False,
        )

        y_pred_train = self.model.predict(dtrain_full)
        self.train_auc = roc_auc_score(y_train, y_pred_train)

        dtest = xgb.DMatrix(X_test)
        self.y_pred_test = self.model.predict(dtest)
        self.test_auc = roc_auc_score(y_test, self.y_pred_test)

        self.test_base_rate = y_test.mean()

        metrics_df = self.create_metrics_df(
            y_test, self.y_pred_test, self.test_base_rate
        )

        performance_df = pd.DataFrame(
            {
                "metric": ["Train AUC", "CV Val AUC", "Test AUC"],
                "value": [self.train_auc, self.cv_val_auc, self.test_auc],
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
            "test_base_rate": self.test_base_rate,
            "best_features_df": self.best_features_df,
        }

    def create_metrics_df(self, y_test, y_pred_test, test_base_rate):
        percentiles = [100, 99] + list(range(95, 0, -5)) + [1, 0]
        table_rows = []
        for p in percentiles:
            cutoff = np.percentile(y_pred_test, p)
            y_pred_binary = (y_pred_test >= cutoff).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
            precision = precision_score(y_test, y_pred_binary, zero_division=0)
            recall = recall_score(y_test, y_pred_binary, zero_division=0)
            f1 = f1_score(y_test, y_pred_binary, zero_division=0)
            accuracy = accuracy_score(y_test, y_pred_binary)
            lift = (
                precision / test_base_rate
                if test_base_rate > 0 and precision > 0
                else 0
            )
            table_rows.append(
                {
                    "percentile": f"P{p}",
                    "cutoff_prob": round(cutoff, 4),
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tn": int(tn),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "accuracy": round(accuracy, 4),
                    "lift": round(lift, 2),
                }
            )
        metrics_df = pd.DataFrame(table_rows).set_index("percentile")
        return metrics_df


# Create synthetic dataset using generator
generator = SyntheticTabularDataDfGenerator()
tabular_data_df = generator.generate("binary:logistic")
print("=== tabular_data_df (first 10 rows) ===")
print(tabular_data_df.head(10))

# Hardcoded configurations (update these based on results from the maximizer script)
target = "target"
selected_features = ["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"]

# Hardcoded configurations (update these based on results from the maximizer script)
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

splitter = TrainTestSplitter(
    tabular_data_df, selected_features, target="target", xgb_objective="binary:logistic"
)
train_df, test_df = splitter.random_split(test_size=0.2, random_state=42)

builder = ModelBuilder(train_df, test_df, selected_features, target, params)
results = builder.build()
model = results["model"]
metrics_df = results["metrics_df"]
performance_df = results["performance_df"]
test_base_rate = results["test_base_rate"]
best_features_df = results["best_features_df"]

# Modify for printing
performance_df["metric"] = performance_df["metric"] + ":"

# Outputs
print("\n=== Model Performance ===")
print(performance_df.to_string())

print(f"\nBase conversion rate on test set: {test_base_rate:.4f}")

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
print(f"Predicted probability: {example_pred:.4f}")

print("\nNOTE:")
print("- 'Pxx' means selecting users with predicted probability >= xx-th percentile")
print("- Higher percentile = stricter threshold = higher precision, lower recall")
print("- Lift = precision / base_rate")
