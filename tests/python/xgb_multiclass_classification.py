# ~/Apps/rtutor/tests/python/xgb_multiclass_classification.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold
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
        self.n_classes = self.train_df[self.target].nunique()
        self.best_features_df = None

    def _compute_cv_auc_best_iter(self, X, y):
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )
        aucs = []
        best_iters = []
        for train_idx, val_idx in skf.split(X, y):
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
            auc = roc_auc_score(y_v, pred_v, multi_class="ovr")
            aucs.append(auc)
            best_iters.append(model.best_iteration)
        return np.mean(aucs), int(np.mean(best_iters))

    def build(self):
        X_train = self.train_df[self.selected_features]
        y_train = self.train_df[self.target]
        X_test = self.test_df[self.selected_features]
        y_test = self.test_df[self.target]

        self.cv_val_auc, best_iter = self._compute_cv_auc_best_iter(X_train, y_train)

        dtrain_full = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)

        self.model = xgb.train(
            self.params,
            dtrain_full,
            num_boost_round=best_iter + 1,
            verbose_eval=False,
        )

        y_pred_train = self.model.predict(dtrain_full)
        self.train_auc = roc_auc_score(y_train, y_pred_train, multi_class="ovr")

        dtest = xgb.DMatrix(X_test)
        self.y_pred_test = self.model.predict(dtest)
        self.test_auc = roc_auc_score(y_test, self.y_pred_test, multi_class="ovr")

        self.test_base_rate = y_test.value_counts(normalize=True).sort_index().to_dict()

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
        results = []
        max_probs = np.max(y_pred_test, axis=1)
        preds_argmax = np.argmax(y_pred_test, axis=1)
        for p in percentiles:
            cutoff = np.percentile(max_probs, p)
            confident_mask = max_probs >= cutoff
            num_classified = np.sum(confident_mask)
            percentile_dict = {}
            percentile_dict["cutoff_prob"] = round(cutoff, 4)
            if num_classified > 0:
                y_test_conf = y_test[confident_mask]
                preds_conf = preds_argmax[confident_mask]
                cm_conf = confusion_matrix(
                    y_test_conf, preds_conf, labels=range(self.n_classes)
                )
                for i in range(self.n_classes):
                    for j in range(self.n_classes):
                        percentile_dict[f"a{i}p{j}"] = cm_conf[i, j]
                precisions_conf = []
                recalls_conf = []
                f1s_conf = []
                for i in range(self.n_classes):
                    tp = cm_conf[i, i]
                    fp = np.sum(cm_conf[:, i]) - tp
                    fn = np.sum(cm_conf[i, :]) - tp
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = (
                        2 * precision * recall / (precision + recall)
                        if (precision + recall) > 0
                        else 0.0
                    )
                    precisions_conf.append(precision)
                    recalls_conf.append(recall)
                    f1s_conf.append(f1)
                precision_macro = np.mean(precisions_conf)
                recall_macro = np.mean(recalls_conf)
                f1_macro = np.mean(f1s_conf)
                accuracy = accuracy_score(y_test_conf, preds_conf)
            else:
                for i in range(self.n_classes):
                    for j in range(self.n_classes):
                        percentile_dict[f"a{i}p{j}"] = 0
                precisions_conf = [0.0] * self.n_classes
                recalls_conf = [0.0] * self.n_classes
                f1s_conf = [0.0] * self.n_classes
                precision_macro = 0.0
                recall_macro = 0.0
                f1_macro = 0.0
                accuracy = 0.0
            percentile_dict["macro_precision"] = round(precision_macro, 4)
            percentile_dict["macro_recall"] = round(recall_macro, 4)
            percentile_dict["macro_f1"] = round(f1_macro, 4)
            percentile_dict["accuracy"] = round(accuracy, 4)
            lifts_conf = [
                precisions_conf[i] / test_base_rate[i]
                if test_base_rate[i] > 0 and precisions_conf[i] > 0
                else 0
                for i in range(self.n_classes)
            ]
            for i in range(self.n_classes):
                percentile_dict[f"c{i}_precision"] = round(precisions_conf[i], 4)
                percentile_dict[f"c{i}_recall"] = round(recalls_conf[i], 4)
                percentile_dict[f"c{i}_f1"] = round(f1s_conf[i], 4)
                percentile_dict[f"c{i}_lift"] = round(lifts_conf[i], 2)
            results.append(percentile_dict)

        metrics_df = pd.DataFrame(results, index=[f"P{p}" for p in percentiles])
        return metrics_df


# Create synthetic dataset using generator
generator = SyntheticTabularDataDfGenerator()
tabular_data_df = generator.generate("multi:softprob")
print("=== tabular_data_df (first 10 rows) ===")
print(tabular_data_df.head(10))

# Hardcoded configurations (update these based on results from the maximizer script)
target = "target"
selected_features = ["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"]

params = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
params["num_class"] = tabular_data_df[target].nunique()

splitter = TrainTestSplitter(
    tabular_data_df, selected_features, target="target", xgb_objective="multi:softprob"
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
print(performance_df.to_string(index=False))

print("\nClass base rates on test set:")
print(test_base_rate)

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
print(f"Predicted class probabilities: {example_pred}")
print(f"Predicted class: {np.argmax(example_pred)}")

print("\nNOTE:")
print(
    "- 'Pxx' means selecting samples with max predicted probability >= xx-th percentile"
)
print(
    "- Higher percentile = stricter threshold = higher macro_precision, lower coverage"
)
print("- Lift = precision / base_rate for each class")
