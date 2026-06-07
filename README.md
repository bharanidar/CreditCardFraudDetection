# Credit Card Fraud Detection System

A simple, beginner-friendly project that demonstrates how to build a machine learning pipeline for detecting fraudulent credit card transactions. The project includes data exploration, model training (Random Forest), model persistence with Joblib, and a Flask REST API for real-time inference.

## Features

- Exploratory Data Analysis (EDA) to understand transaction patterns and class imbalance
- Data preprocessing and feature preparation
- Training a Random Forest classifier to detect fraudulent transactions
- Model persistence using Joblib (`models/fraud_model.pkl`)
- Flask REST API for serving predictions in real time
- Visualizations using Matplotlib (e.g., fraud vs non-fraud distribution)

## Tech Stack

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Joblib

## Dataset

This project uses the popular Credit Card Fraud Detection dataset which contains 284,807 transactions. The dataset is highly imbalanced — the majority of transactions are legitimate and a very small portion are fraudulent. The dataset typically includes features derived from PCA (V1..V28), `Time`, `Amount`, and the target column `Class` (0 = Non-Fraud, 1 = Fraud).

Source: Public benchmark dataset (place `creditcard.csv` under `dataset/` folder).

## Note about the dataset

The dataset is not included in this repository due to GitHub file size limits. The original Credit Card Fraud Detection dataset (approx. 284,807 transactions) can be downloaded from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading, place the `creditcard.csv` file inside the `dataset/` folder before running the training script or the API.

## Model Performance

- Model: Random Forest Classifier
- Reported Accuracy: 99%+ (note: accuracy can be misleading on imbalanced data; monitor precision/recall and use metrics like AUC-ROC)

## API Endpoints

- GET `/` — Health check. Returns a simple message confirming the API is running.
- POST `/predict` — Accepts JSON payload with transaction features and returns prediction:
  - `{ "prediction": "Fraud" }` or `{ "prediction": "Not Fraud" }`

Example request body (single transaction):

```
{
  "Time": 12345,
  "V1": -1.3598071336738,
  "V2": -0.0727811733098497,
  ...
}
```

## Project Structure

```
CreditCardFraudDetection/
├── app.py               # Flask REST API
├── train_model.py       # Training script (EDA, model training, save model)
├── README.md            # Project documentation
├── requirements.txt     # Project dependencies
├── dataset/             # Place creditcard.csv here
├── models/              # Saved model and generated charts
└── notebooks/           # Optional notebooks for EDA and experiments
```

## Installation

1. Clone the repository

```bash
git clone <repo_url>
cd CreditCardFraudDetection
```

2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Place the `creditcard.csv` dataset inside the `dataset/` folder.

4. Train the model

```bash
python train_model.py
```

5. Run the API

```bash
python app.py
```

The API will be available at `http://localhost:5001/`.

## Future Enhancements

- Add data preprocessing steps and feature engineering
- Use cross-validation and hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Implement more evaluation metrics (AUC-ROC, F1-score) and robust logging
- Add authentication to the API and input validation schema (e.g., using Marshmallow or Pydantic)
- Deploy the API using Docker and a production-grade WSGI server (Gunicorn)
- Add unit tests and CI/CD pipeline

---

Contributions welcome — feel free to open issues or pull requests to improve the project.
