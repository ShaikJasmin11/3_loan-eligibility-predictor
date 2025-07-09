#  Loan Eligibility Predictor

> RISE Internship Project 3 – Tamizhan Skills  
> Built with Scikit-Learn, Logistic Regression, and Streamlit

A machine learning-based web app that predicts whether a loan application should be approved based on the applicant’s income, education, credit history, and other financial factors. This is the third project from the **Machine Learning & AI** track of the RISE Internship by Tamizhan Skills.

---

##  Project Objective

To build a loan prediction model that:
  - Loads and preprocesses applicant data from a CSV file
  - Trains a logistic regression classifier to predict loan approval (Y/N)
  - Provides an interactive Streamlit interface for real-time predictions

---

##  Tech Stack

- **Python**
- **Pandas / NumPy**
- **Scikit-learn (Logistic Regression)**
- **Joblib** (for saving models)
- **Streamlit** (for UI)

---

##  Project Structure

```bash
loan-eligibility-predictor/
├── app.py                     # Streamlit frontend for loan prediction
├── main.py                    # Model training and evaluation script
├── requirements.txt           # All required packages
├── data/
│   ├── train.csv              # Training data with labels
│   └── test.csv               # Unlabeled test data
├── models/
│   ├── loan_model.pkl         # Trained logistic regression model
│   └── label_encoders.pkl     # Encoders for categorical variables
├── src/
│   └── preprocess.py          # Preprocessing and encoding logic
└── README.md                  # You're reading it 😉
```

---

## Dataset

- Source: Loan Prediction Dataset – Analytics Vidhya(https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- Contains applicant details such as:
- Gender, Marital Status, Education, Income
- Loan Amount, Loan Term, Credit History, etc.
- Label: Loan_Status (Y/N)

---

## How to Run

- Step 1: Install Dependencies
  
```bash
  pip install -r requirements.txt
```

- Step 2: Train the Model
  
```bash
  python main.py
```

- Step 3: Launch the Web App
  
```bash
  streamlit run app.py
```

  ---

## Model Performance

✅ Accuracy: ~80–85% on test split
✅ Simple, interpretable logistic regression model
✅ Trained on clean, preprocessed real-world data

---

## Highlights

- Handles missing values and encodes categorical features
- Trained using a classic supervised ML approach (Logistic Regression)
- Modular structure for maintainability and scalability
- Clean and user-friendly interface using Streamlit
- Supports real-time prediction from manual input

---

## Acknowledgements

Thanks to Tamizhan Skills for the RISE Internship opportunity.

Inspired by real-world banking and finance risk assessment.

Built by @ShaikJasmin11
