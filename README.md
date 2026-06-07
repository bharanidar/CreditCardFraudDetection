# Credit Card Fraud Detection System

## Overview

This project is a Machine Learning-based Credit Card Fraud Detection System built using Python, Scikit-learn, Flask, Pandas, NumPy, and Matplotlib.

The system analyzes transaction data and predicts whether a transaction is fraudulent or legitimate. A Flask REST API is provided for real-time prediction.

---

## Features

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Fraud vs Non-Fraud visualization
- Random Forest Classification Model
- Model persistence using Joblib
- Flask REST API for predictions
- Real-time fraud detection

---

## Tech Stack

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Joblib
- Git & GitHub

---

## Dataset

Dataset: Credit Card Fraud Detection Dataset

Total Records: 284,807

Due to GitHub file size limitations, the dataset is not included in this repository.

Download it from:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the file inside:

`dataset/creditcard.csv`

---

## Model Performance

Random Forest Classifier

- Accuracy: 99.96%
- Precision: 94.12%
- Recall: 81.63%

---

## API Endpoints

### Health Check

`GET /`

Response:

```json
{
  "message": "Credit Card Fraud Detection API Running"
}
```

### Prediction

`POST /predict`

Returns:

```json
{
  "prediction": "Fraud"
}
```

or

```json
{
  "prediction": "Not Fraud"
}
```

---

## Project Structure

```
CreditCardFraudDetection/

├── app.py

├── train_model.py

├── requirements.txt

├── README.md

├── models/

│   ├── fraud_model.pkl

│   └── fraud_vs_nonfraud.png

└── dataset/
```

---

## Installation

1. Clone the repository

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Download dataset and place it in `dataset/`

4. Train model

   ```bash
   python train_model.py
   ```

5. Run Flask API

   ```bash
   python app.py
   ```

---

## Future Improvements

- MySQL integration for prediction logging
- Model monitoring dashboard
- Deployment on cloud platforms
- Advanced fraud detection models

---

## Author

Bharanidar G
