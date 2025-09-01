import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pipelines'))

from model_building import XGBoostModelBuilder
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator

from config import get_model_config

from data_pipeline import data_pipeline
from config import get_data_paths

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def training_pipeline(data_path: str = "data/raw/ChurnModelling.csv", model_params: Optional[Dict[str, Any]] = None,
                      test_size: float = 0.2, random_state: int = 42,
                      model_path: str = "artifacts/models/churn_analysis.joblib"):
    if (not os.path.exists(get_data_paths()['X_train'])) or \
            (not os.path.exists(get_data_paths()['X_test'])) or \
            (not os.path.exists(get_data_paths()['Y_train'])) or \
            (not os.path.exists(get_data_paths()['Y_test'])):

        data_pipeline()
    else:
        logger.info("Loading Data Artifacts from Data Pipeline.")

    X_train = pd.read_csv(get_data_paths()['X_train'])
    Y_train = pd.read_csv(get_data_paths()['Y_train'])
    X_test = pd.read_csv(get_data_paths()['X_test'])
    Y_test = pd.read_csv(get_data_paths()['Y_test'])

    model_builder = XGBoostModelBuilder(**model_params)
    model = model_builder.build_model()

    trainer = ModelTrainer()
    model, _ = trainer.train(model, X_train=X_train, Y_train=Y_train.squeeze())

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    trainer.save_model(model, model_path)

    evaluator = ModelEvaluator(model, "XGBoost")
    evaluation_matrices = evaluator.evaluate(X_test=X_test, Y_test=Y_test)

    for key, value in evaluation_matrices.items():
        if key == "cm":
            print(f"Confusion Matrix:\n {value}")
            continue
        print(f"{key.capitalize()} Score: {value}")


if __name__ == "__main__":
    model_config = get_model_config()
    training_pipeline(model_params=model_config.get("model_params"))
