.PHONY: all clean install train-pipeline data-pipeline streaming-inference run-all help

# Python inside your .venv
VENV_PYTHON = .venv\Scripts\python.exe

all: help

help:
	@echo Available targets:
	@echo "  make install             - Create virtual env & install dependencies"
	@echo "  make data-pipeline       - Run the data pipeline"
	@echo "  make train-pipeline      - Run the training pipeline"
	@echo "  make streaming-inference - Run streaming inference pipeline"
	@echo "  make run-all             - Run all pipelines in sequence"
	@echo "  make clean               - Clean up generated artifacts"

install:
	@echo Creating virtual environment...
	@python -m venv .venv
	@echo Installing dependencies...
	@.venv\Scripts\python.exe -m pip install --upgrade pip
	@.venv\Scripts\python.exe -m pip install -r requirements.txt
	@echo Installation complete!

clean:
	@echo Cleaning up artifacts...
	@if exist artifacts\models rmdir /s /q artifacts\models
	@if exist artifacts\evaluation rmdir /s /q artifacts\evaluation
	@if exist artifacts\predictions rmdir /s /q artifacts\predictions
	@if exist data\processed rmdir /s /q data\processed
	@if exist pipeline.log del /q pipeline.log
	@echo Cleanup done!

data-pipeline:
	@echo Running data pipeline...
	@$(VENV_PYTHON) pipelines\data_pipeline.py

train-pipeline:
	@echo Running training pipeline...
	@$(VENV_PYTHON) pipelines\training_pipeline.py

streaming-inference:
	@echo Running streaming inference pipeline...
	@$(VENV_PYTHON) pipelines\streaming_inference_pipeline.py

run-all:
	@echo Running all pipelines in order...
	@$(MAKE) data-pipeline
	@$(MAKE) train-pipeline
	@$(MAKE) streaming-inference
