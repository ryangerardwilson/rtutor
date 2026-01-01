# ~/Apps/rtutor/tests/python/synthetic_tabular_data_generator.py

import pandas as pd
import numpy as np

class SyntheticTabularDataDfGenerator:

    def generate(self, objective, is_snapshot=False, n_samples=10000, n_features=20, seed=42, start_date='2023-01-01', freq='min'):
        np.random.seed(seed)
        features = [f"feat_{i}" for i in range(n_features)]
        
        if objective == 'binary:logistic':
            X = pd.DataFrame(
                np.random.dirichlet(np.ones(n_features), size=n_samples),
                columns=features,
                index=pd.RangeIndex(n_samples, name="user_id"),
            )
            logits = (
                -3.0
                + 8 * X["feat_0"]
                + 5 * X["feat_1"]
                + np.random.normal(0, 1, n_samples)
            )
            probs = 1 / (1 + np.exp(-logits))
            y = pd.Series(np.random.binomial(1, probs), index=X.index, name="converted")
            target = 'target'
        
        elif objective == 'reg:squarederror':
            X = pd.DataFrame(
                np.random.normal(0, 1, size=(n_samples, n_features)),
                columns=features,
                index=pd.RangeIndex(n_samples, name="id"),
            )
            true_intercept = 1.0
            true_coeffs = np.zeros(n_features)
            true_coeffs[0] = 0.5
            true_coeffs[1] = -0.3
            true_coeffs[2] = 0.2
            true_coeffs[3] = 0.15
            noise = np.random.normal(0, 0.2, n_samples)
            linear = true_intercept + X @ true_coeffs + noise
            y = pd.Series(np.exp(linear), name="target")
            target = 'target'
        
        elif objective == 'multi:softprob':
            n_classes = 3
            X = pd.DataFrame(
                np.random.dirichlet(np.ones(n_features), size=n_samples),
                columns=features,
                index=pd.RangeIndex(n_samples, name="user_id"),
            )
            logits0 = -2 + 5 * X["feat_0"] + 3 * X["feat_1"] + np.random.normal(0, 1, n_samples)
            logits1 = -1 + 4 * X["feat_2"] + 2 * X["feat_3"] + np.random.normal(0, 1, n_samples)
            logits2 = 0 + 3 * X["feat_4"] + 6 * X["feat_5"] + np.random.normal(0, 1, n_samples)
            logits = np.stack([logits0, logits1, logits2], axis=1)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            y = pd.Series([np.random.choice(n_classes, p=probs[i]) for i in range(n_samples)], index=X.index, name="class")
            target = 'target'
        
        else:
            raise ValueError(f"Unsupported objective: {objective}")
        
        tabular_data_df = X.copy()
        tabular_data_df[target] = y
        if not is_snapshot:
            tabular_data_df['timestamp'] = pd.date_range(start=start_date, periods=n_samples, freq=freq)
        
        return tabular_data_df
