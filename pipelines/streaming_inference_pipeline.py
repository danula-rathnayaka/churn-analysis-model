import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_inference import ModelInference

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

inference = ModelInference('artifacts/models/churn_analysis.joblib')


def streaming_inference(inference, data):
    inference.load_encoders('artifacts/encode')

    pred = inference.predict(data)
    return pred


if __name__ == "__main__":
    df = {
        "RowNumber": 1,
        "CustomerId": 15634602,
        "Firstname": "Grace",
        "Lastname": "Williams",
        "CreditScore": 619,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 2,
        "Balance": 0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 101348.88,
    }

    print(streaming_inference(inference, df))
