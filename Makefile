.PHONY: all clean install train-pipeline data-pipeline streaming-inference run-all help data-pipeline-rebuild mlflow-ui stop-all

# Default Python interpreter and venv
PYTHON = python
VENV = .venv\Scripts\activate
VENV_PYTHON = .venv\Scripts\python.exe
MLFLOW_PORT ?= 5001

# Default target
all: help

# Help target
help:
	@echo Available targets:
	@echo "  make install             - Install project dependencies and set up environment"
	@echo "  make data-pipeline       - Run the data pipeline"
	@echo "  make data-pipeline-rebuild - Force rebuild in data pipeline"
	@echo "  make train-pipeline      - Run the training pipeline"
	@echo "  make streaming-inference - Run the streaming inference pipeline with the sample JSON"
	@echo "  make run-all             - Run all pipelines in sequence"
	@echo "  make mlflow-ui           - Launch MLflow UI"
	@echo "  make stop-all            - Stop MLflow server processes"
	@echo "  make clean               - Clean up artifacts"

# Install project dependencies and set up environment
install:
	@echo Installing project dependencies and setting up environment...
	@echo Creating virtual environment...
	@$(PYTHON) -m venv .venv
	@echo Activating virtual environment and installing dependencies...
	@$(VENV_PYTHON) -m pip install --upgrade pip
	@$(VENV_PYTHON) -m pip install -r requirements.txt
	@echo Installation completed successfully!
	@echo To activate the virtual environment, run: call .venv\Scripts\activate

# Clean up
clean:
	@echo Cleaning up artifacts...
	@if exist artifacts\models rmdir /s /q artifacts\models
	@if exist artifacts\evaluation rmdir /s /q artifacts\evaluation
	@if exist artifacts\predictions rmdir /s /q artifacts\predictions
	@if exist artifacts\encode rmdir /s /q artifacts\encode
	@if exist mlruns rmdir /s /q mlruns
	@echo Cleanup completed!

# Run data pipeline
data-pipeline:
	@echo Start running data pipeline...
	@call $(VENV) && $(PYTHON) pipelines\data_pipeline.py
	@echo Data pipeline completed successfully!

# Run data pipeline with force rebuild
data-pipeline-rebuild:
	@echo Running data pipeline with force rebuild...
	@call $(VENV) && $(PYTHON) -c "from pipelines.data_pipeline import data_pipeline; data_pipeline(force_rebuild=True)"

# Run training pipeline
train-pipeline:
	@echo Running training pipeline...
	@call $(VENV) && $(PYTHON) pipelines\training_pipeline.py

# Run streaming inference pipeline with sample JSON
streaming-inference:
	@echo Running streaming inference pipeline with sample JSON...
	@call $(VENV) && $(PYTHON) pipelines\streaming_inference_pipeline.py

# Run all pipelines in sequence
run-all:
	@echo Running all pipelines in sequence...
	@echo ========================================
	@echo Step 1: Running data pipeline
	@echo ========================================
	@call $(VENV) && $(PYTHON) pipelines\data_pipeline.py
	@echo.
	@echo ========================================
	@echo Step 2: Running training pipeline
	@echo ========================================
	@call $(VENV) && $(PYTHON) pipelines\training_pipeline.py
	@echo.
	@echo ========================================
	@echo Step 3: Running streaming inference pipeline
	@echo ========================================
	@call $(VENV) && $(PYTHON) pipelines\streaming_inference_pipeline.py
	@echo.
	@echo ========================================
	@echo All pipelines completed successfully!
	@echo ========================================

# Launch MLflow UI
mlflow-ui:
	@echo Launching MLflow UI...
	@echo MLflow UI will be available at: http://localhost:$(MLFLOW_PORT)
	@echo Press Ctrl+C to stop the server
	@cmd /c start "" $(VENV_PYTHON) -m mlflow ui --host 0.0.0.0 --port $(MLFLOW_PORT)

# Stop all running MLflow servers
stop-all:
	@echo Stopping all MLflow servers...
	@echo Finding MLflow processes on port $(MLFLOW_PORT)...
	@for /f "tokens=5" %%a in ('netstat -aon ^| findstr :$(MLFLOW_PORT)') do taskkill /PID %%a /F >nul 2>&1
	@echo Finding other MLflow UI processes...
	@for /f "tokens=2" %%a in ('tasklist ^| findstr mlflow') do taskkill /PID %%a /F >nul 2>&1
	@echo ✅ All MLflow servers have been stopped
