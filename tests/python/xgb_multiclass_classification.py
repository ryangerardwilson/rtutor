# ~/Apps/rtutor/tests/python/xgb_multiclass_classification.py
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)
from sklearn.feature_selection import RFE
from xgb_train_test_splitter import TrainTestSplitter

# Set seed for reproducibility
np.random.seed(42)

# Create dummy dataset
n_users = 10000
n_features = 20
n_classes = 3

features = [f"feat_{i}" for i in range(n_features)]
X = pd.DataFrame(
    np.random.dirichlet(np.ones(n_features), size=n_users),
    columns=features,
    index=pd.RangeIndex(n_users, name="user_id"),
)

# Synthetic multi-class target correlated with a few features
logits0 = -2 + 5 * X["feat_0"] + 3 * X["feat_1"] + np.random.normal(0, 1, n_users)
logits1 = -1 + 4 * X["feat_2"] + 2 * X["feat_3"] + np.random.normal(0, 1, n_users)
logits2 = 0 + 3 * X["feat_4"] + 6 * X["feat_5"] + np.random.normal(0, 1, n_users)
logits = np.stack([logits0, logits1, logits2], axis=1)
exp_logits = np.exp(logits)
probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
y = pd.Series([np.random.choice(n_classes, p=probs[i]) for i in range(n_users)], index=X.index, name="class")

# Combine features and target into one DataFrame before modeling
tabular_data_df = X.copy()
tabular_data_df["class"] = y
tabular_data_df['timestamp'] = pd.date_range(start='2023-01-01', periods=n_users, freq='min')

print("=== tabular_data_df (first 10 rows) ===")
print(tabular_data_df.head(10))

class AUCMaximizer:
    def __init__(self, train_df, test_df, features, target):
        self.train_df = train_df
        self.test_df = test_df
        self.features = features
        self.target = target
        self.n_classes = self.train_df[self.target].nunique()
        self.n_features_to_select = 10
        self.n_trials = 30
        self.val_size = 0.2
        self.random_state = 42
        self.default_params = {
            'objective': 'multi:softprob',
            'num_class': self.n_classes,
            'eval_metric': 'mlogloss',
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
        self.num_boost_round = 200
        self.early_stopping_rounds = 20
        self.optuna_boost_round = 1000
        self.optuna_early_stopping = 20
        self.optuna_search_spaces = {
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'eta': {'low': 0.01, 'high': 0.3, 'log': True},
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
        self.results = {}
        self.comparative_df = None
        self.best_name = None
        self.best_auc = None
        self.model = None
        self.X_test_selected = None
        self.y_test = None
        self.selected_features = None
        self.y_pred_test = None
        self.test_base_rate = None
        self.best_features_df = None

    def manual_without_rfe(self):
        X_train = self.train_df[self.features]
        y_train = self.train_df[self.target]
        X_test = self.test_df[self.features]
        y_test = self.test_df[self.target]
        
        selected_features = self.features  # No RFE, use all features
        
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
        
        params = self.default_params.copy()
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=[(dtest, 'eval')],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False,
        )
        y_pred_train = model.predict(dtrain)
        train_auc = roc_auc_score(y_train, y_pred_train, multi_class='ovr')
        return model, X_test, y_test, selected_features, train_auc

    def manual_with_rfe(self):
        X_train = self.train_df[self.features]
        y_train = self.train_df[self.target]
        X_test = self.test_df[self.features]
        y_test = self.test_df[self.target]
        
        # RFE for feature selection
        base_model = xgb.XGBClassifier(
            **self.default_params,
            random_state=self.random_state,
            enable_categorical=True,
        )
        rfe = RFE(estimator=base_model, n_features_to_select=self.n_features_to_select)
        rfe.fit(X_train, y_train)
        
        selected_features = X_train.columns[rfe.support_].tolist()
        
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        dtrain = xgb.DMatrix(X_train_selected, label=y_train, enable_categorical=True)
        dtest = xgb.DMatrix(X_test_selected, label=y_test, enable_categorical=True)
        
        params = self.default_params.copy()
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=[(dtest, 'eval')],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False,
        )
        y_pred_train = model.predict(dtrain)
        train_auc = roc_auc_score(y_train, y_pred_train, multi_class='ovr')
        return model, X_test_selected, y_test, selected_features, train_auc

    def automated_without_rfe(self):
        train_full_df = self.train_df
        X_train_full = train_full_df[self.features]
        y_train_full = train_full_df[self.target]
        X_test = self.test_df[self.features]
        y_test = self.test_df[self.target]

        sub_train_df, val_df = train_test_split(
            train_full_df,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=train_full_df[self.target]
        )

        X_train = sub_train_df[self.features]
        y_train = sub_train_df[self.target]
        X_val = val_df[self.features]
        y_val = val_df[self.target]
        
        selected_features = X_train.columns.tolist()  # No RFE, use all features
        
        # Define the objective function for Optuna
        def objective(trial):
            max_depth = trial.suggest_categorical('max_depth', self.optuna_search_spaces['max_depth'])
            eta = trial.suggest_float('eta', 
                                      self.optuna_search_spaces['eta']['low'], 
                                      self.optuna_search_spaces['eta']['high'], 
                                      log=self.optuna_search_spaces['eta']['log'])
            subsample = trial.suggest_categorical('subsample', self.optuna_search_spaces['subsample'])
            colsample_bytree = trial.suggest_categorical('colsample_bytree', self.optuna_search_spaces['colsample_bytree'])
            params = {
                'objective': 'multi:softprob',
                'num_class': self.n_classes,
                'eval_metric': 'mlogloss',
                'max_depth': max_depth,
                'eta': eta,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
            }
            dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
            dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=self.optuna_boost_round,
                evals=[(dval, 'eval')],
                early_stopping_rounds=self.optuna_early_stopping,
                verbose_eval=False,
            )
            y_pred_val = model.predict(dval)
            auc = roc_auc_score(y_val, y_pred_val, multi_class='ovr')
            return auc
        
        # Create and optimize the study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        # Get the best parameters
        best_params = study.best_params
        best_params['objective'] = 'multi:softprob'
        best_params['num_class'] = self.n_classes
        best_params['eval_metric'] = 'mlogloss'
        
        # Train the final model with best params on full train set
        dtrain_full = xgb.DMatrix(X_train_full, label=y_train_full, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
        
        model = xgb.train(
            best_params,
            dtrain_full,
            num_boost_round=self.optuna_boost_round,
            evals=[(dtest, 'eval')],
            early_stopping_rounds=self.optuna_early_stopping,
            verbose_eval=False,
        )
        y_pred_train = model.predict(dtrain_full)
        train_auc = roc_auc_score(y_train_full, y_pred_train, multi_class='ovr')
        return model, X_test, y_test, selected_features, train_auc

    def automated_with_rfe(self):
        train_full_df = self.train_df
        X_train_full = train_full_df[self.features]
        y_train_full = train_full_df[self.target]
        X_test = self.test_df[self.features]
        y_test = self.test_df[self.target]

        sub_train_df, val_df = train_test_split(
            train_full_df,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=train_full_df[self.target]
        )

        X_train = sub_train_df[self.features]
        y_train = sub_train_df[self.target]
        X_val = val_df[self.features]
        y_val = val_df[self.target]
        
        # RFE for feature selection
        base_model = xgb.XGBClassifier(
            **self.default_params,
            random_state=self.random_state,
            enable_categorical=True,
        )
        rfe = RFE(estimator=base_model, n_features_to_select=self.n_features_to_select)
        rfe.fit(X_train, y_train)
        
        selected_features = X_train.columns[rfe.support_].tolist()
        
        X_train_selected = X_train[selected_features]
        X_val_selected = X_val[selected_features]
        X_train_full_selected = X_train_full[selected_features]
        X_test_selected = X_test[selected_features]
        
        # Define the objective function for Optuna on selected features
        def objective(trial):
            max_depth = trial.suggest_categorical('max_depth', self.optuna_search_spaces['max_depth'])
            eta = trial.suggest_float('eta', 
                                      self.optuna_search_spaces['eta']['low'], 
                                      self.optuna_search_spaces['eta']['high'], 
                                      log=self.optuna_search_spaces['eta']['log'])
            subsample = trial.suggest_categorical('subsample', self.optuna_search_spaces['subsample'])
            colsample_bytree = trial.suggest_categorical('colsample_bytree', self.optuna_search_spaces['colsample_bytree'])
            params = {
                'objective': 'multi:softprob',
                'num_class': self.n_classes,
                'eval_metric': 'mlogloss',
                'max_depth': max_depth,
                'eta': eta,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
            }
            dtrain = xgb.DMatrix(X_train_selected, label=y_train, enable_categorical=True)
            dval = xgb.DMatrix(X_val_selected, label=y_val, enable_categorical=True)
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=self.optuna_boost_round,
                evals=[(dval, 'eval')],
                early_stopping_rounds=self.optuna_early_stopping,
                verbose_eval=False,
            )
            y_pred_val = model.predict(dval)
            auc = roc_auc_score(y_val, y_pred_val, multi_class='ovr')
            return auc
        
        # Create and optimize the study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        # Get the best parameters
        best_params = study.best_params
        best_params['objective'] = 'multi:softprob'
        best_params['num_class'] = self.n_classes
        best_params['eval_metric'] = 'mlogloss'
        
        # Train the final model with best params on full train set with selected features
        dtrain_full = xgb.DMatrix(X_train_full_selected, label=y_train_full, enable_categorical=True)
        dtest = xgb.DMatrix(X_test_selected, label=y_test, enable_categorical=True)
        
        model = xgb.train(
            best_params,
            dtrain_full,
            num_boost_round=self.optuna_boost_round,
            evals=[(dtest, 'eval')],
            early_stopping_rounds=self.optuna_early_stopping,
            verbose_eval=False,
        )
        y_pred_train = model.predict(dtrain_full)
        train_auc = roc_auc_score(y_train_full, y_pred_train, multi_class='ovr')
        return model, X_test_selected, y_test, selected_features, train_auc

    def run_all(self):
        model1, X_test1, y_test, sel1, train_auc1 = self.manual_without_rfe()
        y_pred1 = model1.predict(xgb.DMatrix(X_test1))
        test_auc1 = roc_auc_score(y_test, y_pred1, multi_class='ovr')
        self.results["Manual without RFE"] = (test_auc1, train_auc1, model1, X_test1, y_test, sel1, y_pred1)

        model2, X_test2, y_test, sel2, train_auc2 = self.manual_with_rfe()
        y_pred2 = model2.predict(xgb.DMatrix(X_test2))
        test_auc2 = roc_auc_score(y_test, y_pred2, multi_class='ovr')
        self.results["Manual with RFE"] = (test_auc2, train_auc2, model2, X_test2, y_test, sel2, y_pred2)

        model3, X_test3, y_test, sel3, train_auc3 = self.automated_without_rfe()
        y_pred3 = model3.predict(xgb.DMatrix(X_test3))
        test_auc3 = roc_auc_score(y_test, y_pred3, multi_class='ovr')
        self.results["Automated without RFE"] = (test_auc3, train_auc3, model3, X_test3, y_test, sel3, y_pred3)

        model4, X_test4, y_test, sel4, train_auc4 = self.automated_with_rfe()
        y_pred4 = model4.predict(xgb.DMatrix(X_test4))
        test_auc4 = roc_auc_score(y_test, y_pred4, multi_class='ovr')
        self.results["Automated with RFE"] = (test_auc4, train_auc4, model4, X_test4, y_test, sel4, y_pred4)

    def build_comparative(self):
        comparative_data = {
            'method': list(self.results.keys()),
            'train_auc': [self.results[k][1] for k in self.results],
            'test_auc': [self.results[k][0] for k in self.results],
        }
        self.comparative_df = pd.DataFrame(comparative_data)
        self.comparative_df['abs_delta'] = np.abs(self.comparative_df['train_auc'] - self.comparative_df['test_auc'])
        self.comparative_df = self.comparative_df.sort_values(by='abs_delta', ascending=True).reset_index(drop=True)

    def select_best(self):
        self.best_name = self.comparative_df.iloc[0]['method']
        self.best_auc, _, self.model, self.X_test_selected, self.y_test, self.selected_features, self.y_pred_test = self.results[self.best_name]
        self.test_base_rate = self.y_test.value_counts(normalize=True).sort_index().to_dict()

    def optimize(self):
        self.run_all()
        self.build_comparative()
        self.select_best()
        
        # Compute feature importance for the best model
        importance_gain = self.model.get_score(importance_type="gain")
        if importance_gain:
            total_gain = sum(importance_gain.values())
            normalized_gain = {feat: gain / total_gain for feat, gain in importance_gain.items()}
            self.best_features_df = pd.DataFrame({
                "feature": list(normalized_gain.keys()),
                "importance_gain_normalized": list(normalized_gain.values()),
            }).sort_values(by="importance_gain_normalized", ascending=False)
            self.best_features_df["importance_rank"] = range(1, len(self.best_features_df) + 1)
            self.best_features_df = self.best_features_df.set_index("importance_rank")
        else:
            self.best_features_df = pd.DataFrame()

        model_v_baseline_data = {
            'approach': ['Model', 'Baseline'],
            'auc': [self.best_auc, 0.5]
        }
        model_v_baseline_df = pd.DataFrame(model_v_baseline_data).set_index('approach')
        
        return {
            'comparative_df': self.comparative_df,
            'model': self.model,
            'selected_features': self.selected_features,
            'y_test': self.y_test,
            'y_pred_test': self.y_pred_test,
            'test_base_rate': self.test_base_rate,
            'best_features_df': self.best_features_df,
            'model_v_baseline_df': model_v_baseline_df
        }

class MetricsComputer:
    def __init__(self, y_test, y_pred_test, base_rates=None):
        self.y_test = y_test
        self.y_pred_test = y_pred_test
        self.test_base_rate = base_rates if base_rates is not None else y_test.value_counts(normalize=True).sort_index()
        self.n_classes = len(self.test_base_rate)
        self.preds_argmax = np.argmax(self.y_pred_test, axis=1)

    def compute_metrics(self):
        # Confusion matrix for full set (optional, not used in print)
        cm = confusion_matrix(self.y_test, self.preds_argmax, labels=range(self.n_classes))
        
        # Confusion matrix DF
        cm_df = pd.DataFrame(cm, index=[f"actual_{i}" for i in range(self.n_classes)], columns=[f"pred_{i}" for i in range(self.n_classes)])
        
        # Percentile-based metrics
        percentiles = [100, 99] + list(range(95, 0, -5)) + [1, 0]
        results = []
        max_probs = np.max(self.y_pred_test, axis=1)
        for p in percentiles:
            cutoff = np.percentile(max_probs, p)
            confident_mask = max_probs >= cutoff
            num_classified = np.sum(confident_mask)
            percentile_dict = {}
            percentile_dict['cutoff_prob'] = round(cutoff, 4)
            if num_classified > 0:
                y_test_conf = self.y_test[confident_mask]
                preds_conf = self.preds_argmax[confident_mask]
                cm_conf = confusion_matrix(y_test_conf, preds_conf, labels=range(self.n_classes))
                for i in range(self.n_classes):
                    for j in range(self.n_classes):
                        percentile_dict[f'a{i}p{j}'] = cm_conf[i, j]
                precisions_conf = []
                recalls_conf = []
                f1s_conf = []
                for i in range(self.n_classes):
                    tp = cm_conf[i, i]
                    fp = np.sum(cm_conf[:, i]) - tp
                    fn = np.sum(cm_conf[i, :]) - tp
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
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
                        percentile_dict[f'a{i}p{j}'] = 0
                precisions_conf = [0.0] * self.n_classes
                recalls_conf = [0.0] * self.n_classes
                f1s_conf = [0.0] * self.n_classes
                precision_macro = 0.0
                recall_macro = 0.0
                f1_macro = 0.0
                accuracy = 0.0
            percentile_dict['macro_precision'] = round(precision_macro, 4)
            percentile_dict['macro_recall'] = round(recall_macro, 4)
            percentile_dict['macro_f1'] = round(f1_macro, 4)
            percentile_dict['accuracy'] = round(accuracy, 4)
            lifts_conf = [precisions_conf[i] / self.test_base_rate[i] if self.test_base_rate[i] > 0 and precisions_conf[i] > 0 else 0 for i in range(self.n_classes)]
            for i in range(self.n_classes):
                percentile_dict[f'c{i}_precision'] = round(precisions_conf[i], 4)
                percentile_dict[f'c{i}_recall'] = round(recalls_conf[i], 4)
                percentile_dict[f'c{i}_f1'] = round(f1s_conf[i], 4)
                percentile_dict[f'c{i}_lift'] = round(lifts_conf[i], 2)
            results.append(percentile_dict)

        confidence_metrics_df = pd.DataFrame(results).set_index(pd.Index([f'P{p}' for p in percentiles]))
        
        return cm_df, confidence_metrics_df

# Example usage
splitter = TrainTestSplitter(
    tabular_data_df, 
    features, 
    target='class',
    xgb_objective='multi:softprob'
)
# DO NOT REMOVE THE BELOW COMMENTS
# train_df, test_df = splitter.time_percentile_split(timestamp_col='timestamp', percentile=0.8)
# train_df, test_df = splitter.time_split(timestamp_col='timestamp', split_timestamp='2023-01-04 11:20:00')
train_df, test_df = splitter.random_split(test_size=0.2, random_state=42)

maximizer = AUCMaximizer(train_df, test_df, features, 'class')
results = maximizer.optimize()

print("\n=== Comparative Model Results ===")
print(results['comparative_df'].to_string(index=False))

print("\n=== Model vs Baseline ===")
print(results['model_v_baseline_df'].to_string(float_format="{:.4f}".format))

print("\nSelected Features:")
print(results['selected_features'])

print("\nClass base rates on test set:")
print(results['test_base_rate'])

print("\n=== best_features_df (Top Features by Gain) ===")
print(results['best_features_df'].to_string(float_format="{:.4f}".format))

metrics_comp = MetricsComputer(results['y_test'], results['y_pred_test'], results['test_base_rate'])
cm_df, metrics_df = metrics_comp.compute_metrics()
print("\n=== Performance by Percentile Threshold (Best Model) ===")
print(metrics_df.to_string())

print("\nNOTE:")
print("- Metrics include macro averages and per-class precision, recall, f1, lift")
print("- Lift = precision / base_rate for each class")
print("- 'Pxx' means selecting samples with max predicted probability >= xx-th percentile")
print("- Higher percentile = stricter threshold = higher precision_macro, lower coverage")

# Demonstrate making a prediction
example_row = tabular_data_df.iloc[0][results['selected_features']]
dexample = xgb.DMatrix(pd.DataFrame([example_row]))
example_pred = results['model'].predict(dexample)[0]

print(f"\n=== Example Prediction ===")
print(f"Input features:\n{results['selected_features']}")
print(f"Predicted class probabilities: {example_pred}")
print(f"Predicted class: {np.argmax(example_pred)}")
