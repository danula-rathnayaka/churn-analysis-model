import os
from pathlib import Path

import joblib


class ModelTrainer:
    def train(self, model, X_train, Y_train):
        model.fit(X_train, Y_train)
        train_score = model.score(X_train, Y_train)
        return model, train_score

    def save_model(self, model, file_path: Path):
        joblib.dump(model, file_path)

    def load_model(self, file_path: Path):
        if not os.path.exists(file_path):
            raise ValueError("Model can not be loaded, file not found.")

        return joblib.load(file_path)
