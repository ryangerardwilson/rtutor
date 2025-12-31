# ~/Apps/rtutor/tests/xgb_test_train_split.py

import pandas as pd
from sklearn.model_selection import train_test_split

class TestTrainSplitter:
    def __init__(self, df, features, target, xgb_objective):
        self.df = df
        self.features = features
        self.target = target
        self.xgb_objective = xgb_objective

    def random_split(self, test_size=0.2, random_state=42, stratify=None):
        if stratify is None:
            # Automatically determine stratify based on xgb_objective
            if 'reg:' in self.xgb_objective:
                stratify = False
            else:
                stratify = True  # For classification xgb_objectives like 'binary:logistic'
        
        strat = self.df[self.target] if stratify else None
        df_train, df_test = train_test_split(
            self.df,
            test_size=test_size,
            random_state=random_state,
            stratify=strat
        )
        print(f"Train data rows: {len(df_train)}")
        print(f"Test data rows: {len(df_test)}")
        return df_train, df_test

    def time_split(self, timestamp_col, split_timestamp):
        split_timestamp = pd.to_datetime(split_timestamp)
        train_df = self.df[self.df[timestamp_col] < split_timestamp]
        test_df = self.df[self.df[timestamp_col] >= split_timestamp]
        if len(test_df) < 0.1 * len(train_df):
            raise ValueError("Test data is less than 10% of train data.")
        print(f"Train data rows: {len(train_df)}")
        print(f"Test data rows: {len(test_df)}")
        return train_df, test_df

    def time_percentile_split(self, timestamp_col, percentile=0.8):
        if not 0.01 <= percentile <= 1.00:
            raise ValueError("percentile must be between 0.01 and 1.00")
        df_sorted = self.df.sort_values(by=timestamp_col)
        split_idx = int(len(df_sorted) * percentile)
        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:]
        if len(test_df) < 0.1 * len(train_df):
            raise ValueError("Test data is less than 10% of train data.")
        print(f"Train data rows: {len(train_df)}")
        print(f"Test data rows: {len(test_df)}")
        return train_df, test_df
