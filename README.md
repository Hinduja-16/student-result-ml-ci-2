# Student Result Prediction ML Model with GitHub Actions CI

This project implements an automated Continuous Integration (CI) pipeline for training, evaluating, and testing a Logistic Regression model that predicts student pass/fail results.

## Repository Structure

- `.github/workflows/ml-ci.yml`: GitHub Actions CI pipeline configuration
- `requirements.txt`: Python package dependencies (NumPy, pandas, scikit-learn, joblib)
- `train_model.py`: Dataset generation, feature scaling, model training, and metrics export
- `test_ml_pipeline.py`: Automated unit test suite verifying generated artifacts and predictions

## ML Pipeline Workflow

1. Environment setup and library installation
2. Dataset creation (300 records) and 80/20 train/test split
3. Model training using `StandardScaler` and `LogisticRegression`
4. Automated verification of generated artifacts (`.pkl`, `.json`, `.csv`) and model prediction tests
