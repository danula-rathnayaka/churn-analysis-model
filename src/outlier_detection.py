import logging
from abc import ABC

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class OutlierDetectionStrategy(ABC):
    def detect_outliers(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        pass


class IQROutlierDetectionStrategy(OutlierDetectionStrategy):
    def detect_outliers(self, df: pd.DataFrame, columns: list):
        outliers = pd.DataFrame(False, index=df.index, columns=columns)
        for col in columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
        logging.info("Outliers dected using IQR method")
        return outliers


class OutlierDetector:
    def __init__(self, strategy):
        self._strategy = strategy

    def detect_outliers(self, df, selected_cols):
        return self._strategy.detect_outliers(df, selected_cols)

    def handle_outliers(self, df, selected_cols, method="remove"):
        outliers = self.detect_outliers(df, selected_cols)
        outliers_count = outliers.sum(axis=1)
        rows_to_remove = outliers_count >= 2
        return df[~rows_to_remove]
