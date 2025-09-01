# Customer Churn Prediction Pipeline

A scalable, end-to-end machine learning pipeline to predict customer churn using structured retail banking data. This project automates data preprocessing, model training, evaluation, and experiment tracking with MLflow designed for configurability, reproducibility, and easy extension.

---

## Key Features

* **Data Preparation** - Cleans and transforms raw data by handling missing values, outliers, encoding, scaling, and feature engineering.
* **Model Training** - Supports XGBoost, Random Forest, Logistic Regression, Gradient Boosting, and SVM with configurable hyperparameters.
* **Evaluation & Reporting** - Calculates accuracy, precision, recall, F1-score, and ROC-AUC. Generates confusion matrix, ROC curve, and feature importance plots.
* **Experiment Tracking** - Integrated with MLflow to automatically log parameters, metrics, models, and artifacts for easy experiment comparison.
* **Artifact Management** - Organizes trained models, evaluation reports, visualizations, and metadata in a reproducible format.

---

## Configuration

All settings are controlled through `config/config.yaml`:

* Data paths (input/output/artifacts)
* Feature engineering rules
* Model selection and hyperparameters
* Training strategy and metrics
* MLflow tracking setup

Modify this config file to customize pipeline behavior without changing code.

---

## Getting Started

### Prerequisites

* Python 3.8+
* Git
* Make (or Windows equivalent)

### Setup

```bash
git clone https://github.com/danula-rathnayaka/churn-analysis-model.git
cd churn-analysis-model
make install
```

Create a `.env` file in the root directory with your API key:

```env
GROQ_API_KEY=
```

---

## Usage

Use the provided `Makefile` to run pipeline stages:

| Step                | Description                                    | Command                      |
| ------------------- | ---------------------------------------------- | ---------------------------- |
| Data pipeline       | Preprocess raw data and split into train/test  | `make data-pipeline`         |
| Rebuild data        | Force fresh preprocessing and artifact cleanup | `make data-pipeline-rebuild` |
| Model training      | Train the model and evaluate results           | `make train-pipeline`        |
| Streaming inference | Run real-time predictions on sample data       | `make streaming-inference`   |
| Full pipeline       | Run data, training, and inference pipelines    | `make run-all`               |
| Launch MLflow UI    | Open MLflow tracking dashboard                 | `make mlflow-ui`             |
| Stop MLflow server  | Kill running MLflow processes                  | `make stop-all`              |
| Clean artifacts     | Delete all generated files and logs            | `make clean`                 |

---

## Output

After training, expect the following outputs:

* Processed datasets: `X_train.csv`, `Y_train.csv`, `X_test.csv`, `Y_test.csv`
* Trained model: `churn_analysis.joblib`
* Visualizations: Confusion matrix, ROC curve, feature importance, prediction distributions
* Metadata and training summary reports
* All artifacts logged in MLflow (`mlruns/` directory)
