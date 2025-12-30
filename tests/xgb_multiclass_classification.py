# ~/Apps/rtutor/tests/xgb_multiclass_classification.py
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

class TestTrainSplitter:
    def __init__(self, df, features, target, test_size=0.2, random_state=42, timestamp_col=None, split_timestamp=None):
        self.df = df
        self.features = features
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.timestamp_col = timestamp_col
        self.split_timestamp = pd.to_datetime(split_timestamp) if split_timestamp else None

    def random_split(self, stratify=True):
        strat = self.df[self.target] if stratify else None
        df_train, df_test = train_test_split(
            self.df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=strat
        )
        print(f"Train data rows: {len(df_train)}")
        print(f"Test data rows: {len(df_test)}")
        return df_train, df_test

    def time_split(self):
        if self.split_timestamp is None:
            raise ValueError("split_timestamp must be provided for time_split")
        if self.timestamp_col is None:
            raise ValueError("timestamp_col must be provided for time_split")
        train_df = self.df[self.df[self.timestamp_col] < self.split_timestamp]
        test_df = self.df[self.df[self.timestamp_col] >= self.split_timestamp]
        if len(test_df) < 0.1 * len(train_df):
            raise ValueError("Test data is less than 10% of train data.")
        print(f"Train data rows: {len(train_df)}")
        print(f"Test data rows: {len(test_df)}")
        return train_df, test_df

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
        self.base_rates = None
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
        return model, X_test, y_test, selected_features

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
        print("\n=== Selected Features from RFE (Manual) ===")
        print(selected_features)
        
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
        return model, X_test_selected, y_test, selected_features

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
        
        print('Best hyperparameters found by Optuna (without RFE):')
        print(best_params)
        
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
        return model, X_test, y_test, selected_features

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
        print("\n=== Selected Features from RFE (Automated) ===")
        print(selected_features)
        
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
        
        print('Best hyperparameters found by Optuna (with RFE):')
        print(best_params)
        
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
        return model, X_test_selected, y_test, selected_features

    def run_all(self):
        print("\n=== 1. Manual without RFE ===")
        model1, X_test1, y_test, sel1 = self.manual_without_rfe()
        y_pred1 = model1.predict(xgb.DMatrix(X_test1))
        auc1 = roc_auc_score(y_test, y_pred1, multi_class='ovr')
        print(f"AUC (ovr): {auc1:.4f}")
        self.results["Manual without RFE"] = (auc1, model1, X_test1, y_test, sel1, y_pred1)

        print("\n=== 2. Manual with RFE ===")
        model2, X_test2, y_test, sel2 = self.manual_with_rfe()
        y_pred2 = model2.predict(xgb.DMatrix(X_test2))
        auc2 = roc_auc_score(y_test, y_pred2, multi_class='ovr')
        print(f"AUC (ovr): {auc2:.4f}")
        self.results["Manual with RFE"] = (auc2, model2, X_test2, y_test, sel2, y_pred2)

        print("\n=== 3. Automated (Optuna) without RFE ===")
        model3, X_test3, y_test, sel3 = self.automated_without_rfe()
        y_pred3 = model3.predict(xgb.DMatrix(X_test3))
        auc3 = roc_auc_score(y_test, y_pred3, multi_class='ovr')
        print(f"AUC (ovr): {auc3:.4f}")
        self.results["Automated without RFE"] = (auc3, model3, X_test3, y_test, sel3, y_pred3)

        print("\n=== 4. Automated (Optuna) with RFE ===")
        model4, X_test4, y_test, sel4 = self.automated_with_rfe()
        y_pred4 = model4.predict(xgb.DMatrix(X_test4))
        auc4 = roc_auc_score(y_test, y_pred4, multi_class='ovr')
        print(f"AUC (ovr): {auc4:.4f}")
        self.results["Automated with RFE"] = (auc4, model4, X_test4, y_test, sel4, y_pred4)

    def build_comparative(self):
        comparative_data = {
            'model': list(self.results.keys()),
            'auc': [self.results[k][0] for k in self.results],
            'num_features': [len(self.results[k][4]) for k in self.results]
        }
        self.comparative_df = pd.DataFrame(comparative_data)
        self.comparative_df = self.comparative_df.sort_values(by='auc', ascending=False).reset_index(drop=True)

    def select_best(self):
        self.best_name = self.comparative_df.iloc[0]['model']
        self.best_auc, self.model, self.X_test_selected, self.y_test, self.selected_features, self.y_pred_test = self.results[self.best_name]
        self.base_rates = self.y_test.value_counts(normalize=True).sort_index()

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

        print("\n=== Comparative Model Results ===")
        print(self.comparative_df.to_string(index=False))

        print("\n" + "="*50)
        print(f"BEST MODEL: {self.best_name}")
        print(f"Best Test AUC (ovr): {self.best_auc:.4f}")
        print("="*50)

        print("\nSelected Features:")
        print(self.selected_features)

        print("\nClass base rates on test set:")
        for cls, rate in self.base_rates.items():
            print(f"Class {cls}: {rate:.4f}")

        print(f'\nAUC (ovr) on test set (best model): {self.best_auc:.4f}\n')

        print("\n=== best_features_df (Top Features by Gain) ===")
        print(self.best_features_df.to_string(float_format="{:.4f}".format))

        metrics_comp = MetricsComputer(self.y_test, self.y_pred_test, self.base_rates)
        overall_metrics_df, cm_df, metrics_df = metrics_comp.compute_metrics()
        print("\n=== Overall Performance Metrics (Best Model) ===")
        print(overall_metrics_df.to_string())
        print("\n=== Confusion Matrix (Best Model) ===")
        print(cm_df.to_string())
        print("\n=== Performance by Percentile Threshold (Best Model) ===")
        print(metrics_df.to_string())

        print("\nNOTE:")
        print("- Metrics include macro averages and per-class precision, recall, f1, lift")
        print("- Lift = precision / base_rate for each class")
        print("- 'Pxx' means selecting samples with max predicted probability >= xx-th percentile")
        print("- Higher percentile = stricter threshold = higher precision_macro, lower coverage")
        
        return {
            'comparative_df': self.comparative_df,
            'best_name': self.best_name,
            'best_auc': self.best_auc,
            'model': self.model,
            'X_test_selected': self.X_test_selected,
            'y_test': self.y_test,
            'selected_features': self.selected_features,
            'y_pred_test': self.y_pred_test,
            'base_rates': self.base_rates,
            'best_features_df': self.best_features_df
        }

class MetricsComputer:
    def __init__(self, y_test, y_pred_test, base_rates=None):
        self.y_test = y_test
        self.y_pred_test = y_pred_test
        self.base_rates = base_rates if base_rates is not None else y_test.value_counts(normalize=True).sort_index()
        self.n_classes = len(self.base_rates)
        self.preds_argmax = np.argmax(self.y_pred_test, axis=1)

    def compute_metrics(self):
        # Confusion matrix for full set
        cm = confusion_matrix(self.y_test, self.preds_argmax, labels=range(self.n_classes))
        
        # Custom metrics for full set
        precisions = []
        recalls = []
        f1s = []
        for i in range(self.n_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        
        precision_macro = np.mean(precisions)
        recall_macro = np.mean(recalls)
        f1_macro = np.mean(f1s)
        accuracy = accuracy_score(self.y_test, self.preds_argmax)
        
        lifts = [precisions[i] / self.base_rates[i] if self.base_rates[i] > 0 and precisions[i] > 0 else 0 for i in range(self.n_classes)]
        
        # Build overall table
        table_rows = [{
            "metric": "macro_precision",
            "value": round(precision_macro, 4),
        }, {
            "metric": "macro_recall",
            "value": round(recall_macro, 4),
        }, {
            "metric": "macro_f1",
            "value": round(f1_macro, 4),
        }, {
            "metric": "accuracy",
            "value": round(accuracy, 4),
        }]
        
        for i in range(self.n_classes):
            table_rows.append({
                "metric": f"class_{i}_precision",
                "value": round(precisions[i], 4),
            })
            table_rows.append({
                "metric": f"class_{i}_recall",
                "value": round(recalls[i], 4),
            })
            table_rows.append({
                "metric": f"class_{i}_f1",
                "value": round(f1s[i], 4),
            })
            table_rows.append({
                "metric": f"class_{i}_lift",
                "value": round(lifts[i], 2),
            })
        
        overall_metrics_df = pd.DataFrame(table_rows).set_index('metric')
        
        # Confusion matrix DF
        cm_df = pd.DataFrame(cm, index=[f"actual_{i}" for i in range(self.n_classes)], columns=[f"pred_{i}" for i in range(self.n_classes)])
        
        # Percentile-based metrics
        percentiles = [99] + list(range(95, 0, -5)) + [1]
        results = []
        max_probs = np.max(self.y_pred_test, axis=1)
        for p in percentiles:
            cutoff = np.percentile(max_probs, p)
            confident_mask = max_probs >= cutoff
            num_classified = np.sum(confident_mask)
            if num_classified > 0:
                y_test_conf = self.y_test[confident_mask]
                preds_conf = self.preds_argmax[confident_mask]
                cm_conf = confusion_matrix(y_test_conf, preds_conf, labels=range(self.n_classes))
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
                precision_macro = 0.0
                recall_macro = 0.0
                f1_macro = 0.0
                accuracy = 0.0
            results.append({
                'percentile': f'P{p}',
                'cutoff_prob': round(cutoff, 4),
                'precision_macro': round(precision_macro, 4),
                'recall_macro': round(recall_macro, 4),
                'f1_macro': round(f1_macro, 4),
                'accuracy': round(accuracy, 4),
            })

        confidence_metrics_df = pd.DataFrame(results).set_index('percentile')
        
        return overall_metrics_df, cm_df, confidence_metrics_df

# Example usage
splitter = TestTrainSplitter(tabular_data_df, features, 'class', timestamp_col='timestamp', split_timestamp='2023-01-04 11:20:00')
train_df, test_df = splitter.time_split()
maximizer = AUCMaximizer(train_df, test_df, features, 'class')
results = maximizer.optimize()

# Demonstrate making a prediction
example_row = tabular_data_df.iloc[0][results['selected_features']]
dexample = xgb.DMatrix(pd.DataFrame([example_row]))
example_pred = results['model'].predict(dexample)[0]

print(f"\n=== Example Prediction ===")
print(f"Input features:\n{results['selected_features']}")
print(f"Predicted class probabilities: {example_pred}")
print(f"Predicted class: {np.argmax(example_pred)}")
