# ~/Apps/rtutor/tests/xgb_bin_classification.py
# Updated ~/Apps/rtutor/tests/xgb_bin_classification.py
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from sklearn.feature_selection import RFE
from xgb_train_test_splitter import TrainTestSplitter

# Set seed for reproducibility
np.random.seed(42)

# Create dummy dataset
n_users = 10000
n_features = 20

features = [f"feat_{i}" for i in range(n_features)]
X = pd.DataFrame(
    np.random.dirichlet(np.ones(n_features), size=n_users),
    columns=features,
    index=pd.RangeIndex(n_users, name="user_id"),
)

# Synthetic target correlated with a few features
logits = (
    -3.0
    + 8 * X["feat_0"]
    + 5 * X["feat_1"]
    + np.random.normal(0, 1, n_users)
)
probs = 1 / (1 + np.exp(-logits))
y = pd.Series(np.random.binomial(1, probs), index=X.index, name="converted")

# Combine features and target into one DataFrame before modeling
tabular_data_df = X.copy()
tabular_data_df["converted"] = y
tabular_data_df['timestamp'] = pd.date_range(start='2023-01-01', periods=n_users, freq='min')

print("=== tabular_data_df (first 10 rows) ===")
print(tabular_data_df.head(10))

class AUCMaximizer:
    def __init__(self, train_df, test_df, features, target):
        self.train_df = train_df
        self.test_df = test_df
        self.features = features
        self.target = target
        self.n_features_to_select = 10
        self.n_trials = 30
        self.val_size = 0.2
        self.random_state = 42
        self.default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
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
        self.base_rate = None
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
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
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
            auc = roc_auc_score(y_val, y_pred_val)
            return auc
        
        # Create and optimize the study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        # Get the best parameters
        best_params = study.best_params
        best_params['objective'] = 'binary:logistic'
        best_params['eval_metric'] = 'auc'
        
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
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
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
            auc = roc_auc_score(y_val, y_pred_val)
            return auc
        
        # Create and optimize the study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        # Get the best parameters
        best_params = study.best_params
        best_params['objective'] = 'binary:logistic'
        best_params['eval_metric'] = 'auc'
        
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
        auc1 = roc_auc_score(y_test, y_pred1)
        print(f"AUC: {auc1:.4f}")
        self.results["Manual without RFE"] = (auc1, model1, X_test1, y_test, sel1, y_pred1)

        print("\n=== 2. Manual with RFE ===")
        model2, X_test2, y_test, sel2 = self.manual_with_rfe()
        y_pred2 = model2.predict(xgb.DMatrix(X_test2))
        auc2 = roc_auc_score(y_test, y_pred2)
        print(f"AUC: {auc2:.4f}")
        self.results["Manual with RFE"] = (auc2, model2, X_test2, y_test, sel2, y_pred2)

        print("\n=== 3. Automated (Optuna) without RFE ===")
        model3, X_test3, y_test, sel3 = self.automated_without_rfe()
        y_pred3 = model3.predict(xgb.DMatrix(X_test3))
        auc3 = roc_auc_score(y_test, y_pred3)
        print(f"AUC: {auc3:.4f}")
        self.results["Automated without RFE"] = (auc3, model3, X_test3, y_test, sel3, y_pred3)

        print("\n=== 4. Automated (Optuna) with RFE ===")
        model4, X_test4, y_test, sel4 = self.automated_with_rfe()
        y_pred4 = model4.predict(xgb.DMatrix(X_test4))
        auc4 = roc_auc_score(y_test, y_pred4)
        print(f"AUC: {auc4:.4f}")
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
        self.base_rate = self.y_test.mean()

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
        print(f"Best Test AUC: {self.best_auc:.4f}")
        print("="*50)

        print("\nSelected Features:")
        print(self.selected_features)

        print(f"\nBase conversion rate on test set: {self.base_rate:.4f}")

        print(f'AUC on test set (best model): {self.best_auc:.4f}\n')

        print("\n=== best_features_df (Top Features by Gain) ===")
        print(self.best_features_df.to_string(float_format="{:.4f}".format))

        metrics_comp = MetricsComputer(self.y_test, self.y_pred_test, self.base_rate)
        metrics_df = metrics_comp.compute_metrics()
        print("\n=== Performance by Percentile Threshold (Best Model) ===")
        print(metrics_df.to_string())

        print("\nNOTE:")
        print("- 'Pxx' means selecting users with predicted probability >= xx-th percentile")
        print("- Higher percentile = stricter threshold = higher precision, lower recall")
        print("- Lift = precision / base_rate")
        
        return {
            'comparative_df': self.comparative_df,
            'best_name': self.best_name,
            'best_auc': self.best_auc,
            'model': self.model,
            'X_test_selected': self.X_test_selected,
            'y_test': self.y_test,
            'selected_features': self.selected_features,
            'y_pred_test': self.y_pred_test,
            'base_rate': self.base_rate,
            'best_features_df': self.best_features_df
        }

class MetricsComputer:
    def __init__(self, y_test, y_pred_test, base_rate=None):
        self.y_test = y_test
        self.y_pred_test = y_pred_test
        self.base_rate = base_rate if base_rate is not None else y_test.mean()

    def compute_metrics(self):
        percentiles = [99] + list(range(95, 0, -5)) + [1]
        table_rows = []
        for p in percentiles:
            cutoff = np.percentile(self.y_pred_test, p)
            y_pred_binary = (self.y_pred_test >= cutoff).astype(int)
            tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred_binary).ravel()
            precision = precision_score(self.y_test, y_pred_binary, zero_division=0)
            recall = recall_score(self.y_test, y_pred_binary, zero_division=0)
            f1 = f1_score(self.y_test, y_pred_binary, zero_division=0)
            accuracy = accuracy_score(self.y_test, y_pred_binary)
            lift = precision / self.base_rate if self.base_rate > 0 and precision > 0 else 0
            table_rows.append({
                'percentile': f'P{p}',
                'cutoff_prob': round(cutoff, 4),
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'accuracy': round(accuracy, 4),
                'lift': round(lift, 2),
            })
        metrics_df = pd.DataFrame(table_rows).set_index('percentile')
        return metrics_df

splitter = TrainTestSplitter(
    tabular_data_df, 
    features, 
    target='converted',
    xgb_objective='binary:logistic'
)
# train_df, test_df = splitter.time_percentile_split(timestamp_col='timestamp', percentile=0.8)
# train_df, test_df = splitter.time_split(timestamp_col='timestamp', split_timestamp='2023-01-04 11:20:00')
train_df, test_df = splitter.random_split(test_size=0.2, random_state=42)
maximizer = AUCMaximizer(train_df, test_df, features, 'converted')
results = maximizer.optimize()

# Demonstrate making a prediction
example_row = tabular_data_df.iloc[0][results['selected_features']]
dexample = xgb.DMatrix(pd.DataFrame([example_row]))
example_pred = results['model'].predict(dexample)[0]

print(f"\n=== Example Prediction ===")
print(f"Input features:\n{results['selected_features']}")
print(f"Predicted probability: {example_pred:.4f}")
