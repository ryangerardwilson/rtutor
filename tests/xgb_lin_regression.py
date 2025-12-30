# Updated ~/Apps/rtutor/tests/xgb_lin_regression.py
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.feature_selection import RFE

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
y = np.exp(linear)
y.name = "target"

# Combine into one DataFrame
tabular_data_df = X.copy()
tabular_data_df["target"] = y

print("=== tabular_data_df (first 10 rows) ===")
print(tabular_data_df.head(10))

class R2Maximizer:
    def __init__(self, tabular_data_df, features, target):
        self.tabular_data_df = tabular_data_df
        self.features = features
        self.target = target
        self.n_features_to_select = 10
        self.n_trials = 30
        self.test_size = 0.2
        self.val_size = 0.2
        self.random_state = 42
        self.default_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
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
        self.log_transformation_needed = False
        self.results = {}
        self.comparative_df = None
        self.best_name = None
        self.best_r2 = None
        self.best_mae = None
        self.best_rmse = None
        self.model = None
        self.X_test_selected = None
        self.y_test = None
        self.selected_features = None
        self.y_pred_test = None
        self.y_test_orig = None
        self.y_pred_orig = None
        self.base_mean = None
        self.baseline_r2 = None
        self.baseline_mae = None
        self.baseline_rmse = None
        self.best_features_df = None

        # Skewness Check and log transformation of target
        skew_value = skew(self.tabular_data_df[self.target])
        print(f"\nSkewness of original target: {skew_value:.4f}")
        if abs(skew_value) > 0.5:
            print("Target is skewed. Applying log transformation.")
            self.tabular_data_df[self.target] = np.log(self.tabular_data_df[self.target])
            skew_transformed = skew(self.tabular_data_df[self.target])
            print(f"Skewness after log transformation: {skew_transformed:.4f}")
            self.log_transformation_needed = True
        else:
            print("Target is approximately normal. No transformation applied.")

    def compute_baseline(self):
        X_train, _, y_train, y_test = train_test_split(
            self.tabular_data_df[self.features],
            self.tabular_data_df[self.target],
            test_size=self.test_size,
            random_state=self.random_state,
        )
        y_train_orig = np.exp(y_train) if self.log_transformation_needed else y_train
        y_test_orig = np.exp(y_test) if self.log_transformation_needed else y_test
        mean_target = y_train_orig.mean()
        baseline_pred = np.full_like(y_test_orig, mean_target)
        r2 = r2_score(y_test_orig, baseline_pred)
        mae = mean_absolute_error(y_test_orig, baseline_pred)
        rmse = root_mean_squared_error(y_test_orig, baseline_pred)
        return r2, mae, rmse, mean_target, y_test_orig, baseline_pred

    def manual_without_rfe(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.tabular_data_df[self.features],
            self.tabular_data_df[self.target],
            test_size=self.test_size,
            random_state=self.random_state,
        )
        
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
        X_train, X_test, y_train, y_test = train_test_split(
            self.tabular_data_df[self.features],
            self.tabular_data_df[self.target],
            test_size=self.test_size,
            random_state=self.random_state,
        )
        
        # RFE for feature selection
        base_model = xgb.XGBRegressor(
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
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            self.tabular_data_df.drop(self.target, axis=1),
            self.tabular_data_df[self.target],
            test_size=self.test_size,
            random_state=self.random_state,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=self.val_size,
            random_state=self.random_state,
        )
        
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
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
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
            y_val_orig = np.exp(y_val) if self.log_transformation_needed else y_val
            y_pred_orig = np.exp(y_pred_val) if self.log_transformation_needed else y_pred_val
            r2 = r2_score(y_val_orig, y_pred_orig)
            return r2
        
        # Create and optimize the study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        # Get the best parameters
        best_params = study.best_params
        best_params['objective'] = 'reg:squarederror'
        best_params['eval_metric'] = 'rmse'
        
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
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            self.tabular_data_df.drop(self.target, axis=1),
            self.tabular_data_df[self.target],
            test_size=self.test_size,
            random_state=self.random_state,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=self.val_size,
            random_state=self.random_state,
        )
        
        # RFE for feature selection
        base_model = xgb.XGBRegressor(
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
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
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
            y_val_orig = np.exp(y_val) if self.log_transformation_needed else y_val
            y_pred_orig = np.exp(y_pred_val) if self.log_transformation_needed else y_pred_val
            r2 = r2_score(y_val_orig, y_pred_orig)
            return r2
        
        # Create and optimize the study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        # Get the best parameters
        best_params = study.best_params
        best_params['objective'] = 'reg:squarederror'
        best_params['eval_metric'] = 'rmse'
        
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
        self.baseline_r2, self.baseline_mae, self.baseline_rmse, self.base_mean, _, _ = self.compute_baseline()

        print("\n=== 1. Manual without RFE ===")
        model1, X_test1, y_test, sel1 = self.manual_without_rfe()
        y_pred1 = model1.predict(xgb.DMatrix(X_test1))
        y_test_orig = np.exp(y_test) if self.log_transformation_needed else y_test
        y_pred_orig1 = np.exp(y_pred1) if self.log_transformation_needed else y_pred1
        r21 = r2_score(y_test_orig, y_pred_orig1)
        mae1 = mean_absolute_error(y_test_orig, y_pred_orig1)
        rmse1 = root_mean_squared_error(y_test_orig, y_pred_orig1)
        print(f"R2: {r21:.4f}")
        self.results["Manual without RFE"] = (r21, mae1, rmse1, model1, X_test1, y_test, sel1, y_pred1)

        print("\n=== 2. Manual with RFE ===")
        model2, X_test2, y_test, sel2 = self.manual_with_rfe()
        y_pred2 = model2.predict(xgb.DMatrix(X_test2))
        y_test_orig = np.exp(y_test) if self.log_transformation_needed else y_test
        y_pred_orig2 = np.exp(y_pred2) if self.log_transformation_needed else y_pred2
        r22 = r2_score(y_test_orig, y_pred_orig2)
        mae2 = mean_absolute_error(y_test_orig, y_pred_orig2)
        rmse2 = root_mean_squared_error(y_test_orig, y_pred_orig2)
        print(f"R2: {r22:.4f}")
        self.results["Manual with RFE"] = (r22, mae2, rmse2, model2, X_test2, y_test, sel2, y_pred2)

        print("\n=== 3. Automated (Optuna) without RFE ===")
        model3, X_test3, y_test, sel3 = self.automated_without_rfe()
        y_pred3 = model3.predict(xgb.DMatrix(X_test3))
        y_test_orig = np.exp(y_test) if self.log_transformation_needed else y_test
        y_pred_orig3 = np.exp(y_pred3) if self.log_transformation_needed else y_pred3
        r23 = r2_score(y_test_orig, y_pred_orig3)
        mae3 = mean_absolute_error(y_test_orig, y_pred_orig3)
        rmse3 = root_mean_squared_error(y_test_orig, y_pred_orig3)
        print(f"R2: {r23:.4f}")
        self.results["Automated without RFE"] = (r23, mae3, rmse3, model3, X_test3, y_test, sel3, y_pred3)

        print("\n=== 4. Automated (Optuna) with RFE ===")
        model4, X_test4, y_test, sel4 = self.automated_with_rfe()
        y_pred4 = model4.predict(xgb.DMatrix(X_test4))
        y_test_orig = np.exp(y_test) if self.log_transformation_needed else y_test
        y_pred_orig4 = np.exp(y_pred4) if self.log_transformation_needed else y_pred4
        r24 = r2_score(y_test_orig, y_pred_orig4)
        mae4 = mean_absolute_error(y_test_orig, y_pred_orig4)
        rmse4 = root_mean_squared_error(y_test_orig, y_pred_orig4)
        print(f"R2: {r24:.4f}")
        self.results["Automated with RFE"] = (r24, mae4, rmse4, model4, X_test4, y_test, sel4, y_pred4)

    def build_comparative(self):
        comparative_data = {
            'model': list(self.results.keys()),
            'r2': [self.results[k][0] for k in self.results],
            'mae': [self.results[k][1] for k in self.results],
            'rmse': [self.results[k][2] for k in self.results],
            'num_features': [len(self.results[k][6]) for k in self.results]
        }
        self.comparative_df = pd.DataFrame(comparative_data)
        self.comparative_df = self.comparative_df.sort_values(by='r2', ascending=False).reset_index(drop=True)

    def select_best(self):
        self.best_name = self.comparative_df.iloc[0]['model']
        self.best_r2, self.best_mae, self.best_rmse, self.model, self.X_test_selected, self.y_test, self.selected_features, self.y_pred_test = self.results[self.best_name]
        self.y_test_orig = np.exp(self.y_test) if self.log_transformation_needed else self.y_test
        self.y_pred_orig = np.exp(self.y_pred_test) if self.log_transformation_needed else self.y_pred_test

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
        
        return {
            'comparative_df': self.comparative_df,
            'best_name': self.best_name,
            'best_r2': self.best_r2,
            'best_mae': self.best_mae,
            'best_rmse': self.best_rmse,
            'model': self.model,
            'X_test_selected': self.X_test_selected,
            'y_test': self.y_test,
            'selected_features': self.selected_features,
            'y_pred_test': self.y_pred_test,
            'y_test_orig': self.y_test_orig,
            'y_pred_orig': self.y_pred_orig,
            'base_mean': self.base_mean,
            'baseline_r2': self.baseline_r2,
            'baseline_mae': self.baseline_mae,
            'baseline_rmse': self.baseline_rmse,
            'best_features_df': self.best_features_df
        }

    def print_results(self):
        print("\n=== Comparative Model Results ===")
        print(self.comparative_df.to_string(index=False))

        print("\nBaseline (mean prediction):")
        print(f"R2: {self.baseline_r2:.4f}, MAE: {self.baseline_mae:.4f}, RMSE: {self.baseline_rmse:.4f}")

        print("\n" + "="*50)
        print(f"BEST MODEL: {self.best_name}")
        print(f"Best Test R2: {self.best_r2:.4f}")
        print("="*50)

        print("\nSelected Features:")
        print(self.selected_features)

        print(f"\nBase mean on train set: {self.base_mean:.4f}")

        print(f'R2 on test set (best model): {self.best_r2:.4f}\n')

class MetricsComputer:
    def __init__(self, y_test_orig, y_pred_orig, base_mean):
        self.y_test_orig = y_test_orig
        self.y_pred_orig = y_pred_orig
        self.base_mean = base_mean

    def compute_metrics(self):
        temp = pd.DataFrame({'pred': self.y_pred_orig, 'actual': self.y_test_orig})
        temp['decile'] = pd.qcut(temp['pred'], 10, labels=range(1, 11))
        grouped = temp.groupby('decile', observed=True)
        metrics = grouped.agg(
            count=('pred', 'count'),
            min_pred=('pred', 'min'),
            max_pred=('pred', 'max'),
            avg_pred=('pred', 'mean'),
            avg_actual=('actual', 'mean'),
        )
        metrics['mae'] = grouped.apply(lambda g: mean_absolute_error(g['actual'], g['pred']), include_groups=False)
        metrics['rmse'] = grouped.apply(lambda g: root_mean_squared_error(g['actual'], g['pred']), include_groups=False)
        metrics['lift'] = metrics['avg_actual'] / self.base_mean
        metrics = metrics[['count', 'min_pred', 'max_pred', 'avg_pred', 'avg_actual', 'mae', 'rmse', 'lift']]
        metrics = metrics.sort_index(ascending=False)  # Decile 10 first
        return metrics.round(4)

# Example usage
maximizer = R2Maximizer(tabular_data_df, features, 'target')
results = maximizer.optimize()
maximizer.print_results()

print("\n=== best_features_df (Top Features by Gain) ===")
print(results['best_features_df'].to_string(float_format="{:.4f}".format))

metrics_comp = MetricsComputer(results['y_test_orig'], results['y_pred_orig'], results['base_mean'])
metrics_df = metrics_comp.compute_metrics()
print("\n=== Performance by Decile (Best Model) ===")
print(metrics_df.to_string())

print("\nNOTE:")
print("- 'Decile 10' means top 10% with highest predicted values")
print("- Higher decile = higher precision in predictions, ideally")
print("- Lift = avg_actual / base_mean")

# Demonstrate making a prediction
example_row = maximizer.tabular_data_df.iloc[0][results['selected_features']]
dexample = xgb.DMatrix(pd.DataFrame([example_row]))
example_pred = results['model'].predict(dexample)[0]
example_pred = np.exp(example_pred) if maximizer.log_transformation_needed else example_pred

print(f"\n=== Example Prediction ===")
print(f"Input features:\n{results['selected_features']}")
print(f"Predicted target: {example_pred:.4f}")
