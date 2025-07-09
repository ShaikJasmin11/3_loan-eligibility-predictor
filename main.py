# main.py

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.preprocess import preprocess_data

def train_and_save_model():
    df = pd.read_csv('data/train.csv')
    df.dropna(subset=['Loan_Status'], inplace=True)

    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

    X_raw = df.drop(columns='Loan_Status')
    y = df['Loan_Status']

    X, encoders = preprocess_data(X_raw, is_train=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    joblib.dump(model, 'models/loan_model.pkl')
    joblib.dump(encoders, 'models/label_encoders.pkl')
    print("✅ Model and encoders saved.")

if __name__ == '__main__':
    train_and_save_model()
