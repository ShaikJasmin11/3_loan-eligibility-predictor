# src/preprocess.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df, encoders=None, is_train=True):
    df = df.copy()

    # Fill missing values
    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'Loan_Amount_Term']:
        df[col].fillna(df[col].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

    # Encode categorical columns
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    if is_train:
        encoders = {col: LabelEncoder().fit(df[col]) for col in cat_cols}
    for col in cat_cols:
        df[col] = encoders[col].transform(df[col])

    # Drop Loan_ID if exists
    if 'Loan_ID' in df.columns:
        df.drop(columns=['Loan_ID'], inplace=True)

    return df, encoders
