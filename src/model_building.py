import os
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class BaseModelBuilder(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model = None
        self.model_params = kwargs

    @abstractmethod
    def build_model(self):
        pass

    def save_model(self, file_path: Path):
        if self.model is None:
            raise ValueError("No model to save, build the model first.")

        joblib.dump(self.model, file_path)

    def load_model(self, file_path: Path):
        if os.path.exists(file_path):
            raise ValueError("Model can not be loaded, file not found.")

        joblib.load(file_path)


class RandomForestModelBuilder(BaseModelBuilder):
    def __init__(self, **kwargs):
        default_params = {
            'max_depth': 10,
            'n_estimators': 100,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
        }

        default_params.update(kwargs)
        super().__init__("RandomForestClassifier", **default_params)

    def build_model(self):
        self.model = RandomForestClassifier(self.model_params)
        return self.model


class XGBoostModelBuilder(BaseModelBuilder):
    def __init__(self, **kwargs):
        default_params = {
            'max_depth': 10,
            'n_estimators': 100,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
        }

        default_params.update(kwargs)
        super().__init__("XGBClassifier", **default_params)

    def build_model(self):
        self.model = XGBClassifier(self.model_params)
        return self.model
