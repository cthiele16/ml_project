from sklearn.compose import ColumnTransformer
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

RANDOM_STATE = 20

# Rename X1...X23 with real comumn names
COL_NAMES = {
    "X1": "LIMIT_BAL", "X2": "SEX", "X3": "EDUCATION", "X4": "MARRIAGE",
    "X5": "AGE", "X6": "PAY_0", "X7": "PAY_2", "X8": "PAY_3",
    "X9": "PAY_4", "X10": "PAY_5", "X11": "PAY_6",
    "X12": "BILL_AMT1", "X13": "BILL_AMT2", "X14": "BILL_AMT3",
    "X15": "BILL_AMT4", "X16": "BILL_AMT5", "X17": "BILL_AMT6",
    "X18": "PAY_AMT1", "X19": "PAY_AMT2", "X20": "PAY_AMT3",
    "X21": "PAY_AMT4", "X22": "PAY_AMT5", "X23": "PAY_AMT6",
}

# dataset mentions  SEX, EDUCATION, MARRIAGE as categorical with defined category
# (e.g., EDUCATION: 1=graduate school, 2=university...). 
# PAY_0..PAY_6 are repayment status codes (-2 to 9), not continuous measurements
CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]

# BILL_AMT1-6 = bill amount in month 1/month 2 (one month ago)/...
# PAY_AMT1–6 = how much did the customer actually paid in month 1/month 2 (one month ago)/...
# PAY_0, PAY_2–6 = repayment status each month (PAY_0 is most recent, oddly named):
    # -2 = No consumption (nothing to pay), -1	Paid in full, 0	Minimum payment made (revolving credit), 1	1 month late, 2	2 months late
# So a customer with PAY_0=2, PAY_2=1, PAY_3=0 was 2 months late most recently, 1 month late before that, and paid minimally the month before that

NUMERIC_COLS = ["LIMIT_BAL", "AGE",
                "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
                "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
                "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"
                ]

def load_prepare_data():
    """
    Load dataset, clean variables,
    and create engineered features.
    """

    cache_path = "data/credit_default.csv"

    # download the dataset the first time used and safe it locally afterwards
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
    else:
        # Download dataset
        dataset = fetch_ucirepo(id=350)
        X = dataset.data.features
        y = dataset.data.targets
        # Combine features + target
        df = pd.concat([X, y], axis=1)
        os.makedirs("data", exist_ok=True)
        df.to_csv(cache_path, index=False)

    #Rename Columns
    df = df.rename(columns=COL_NAMES)

    # DATA CLEANING
    # Merge rare education categories
    df["EDUCATION"] = df["EDUCATION"].replace({
        0: 4,
        5: 4,
        6: 4
    })

    # Replace unknown marriage category
    df["MARRIAGE"] = df["MARRIAGE"].replace({
        0: 3
    })

    # Convert sex female = 1, male = 0
    df["SEX"] = (df["SEX"] == 2).astype(int)

    #FEATURE ENGINEERING
     # Credit usage ratio
    df["CREDIT_UTILIZATION"] = (
        df["BILL_AMT1"] / df["LIMIT_BAL"]
    )

    # Payment ratio
    df["PAYMENT_RATIO"] = (
        df["PAY_AMT1"]
        / df["BILL_AMT1"].replace(0, np.nan)
    ).clip(0, 1).fillna(0)

    # Repayment delay columns
    pay_cols = [
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6"
    ]

    # Count delayed months
    df["MONTHS_DELAYED"] = (
        df[pay_cols] > 0
    ).sum(axis=1)

    # Engineered features
    engineered_cols = [
        "CREDIT_UTILIZATION",
        "PAYMENT_RATIO",
        "MONTHS_DELAYED"
    ]

    # Add engineered features to numeric columns
    numeric_cols = NUMERIC_COLS + engineered_cols

    #FEATURES + TARGET
    X = df.drop(columns=["Y"])
    y = df["Y"]

    return X, y, numeric_cols

def create_preprocessor(numeric_cols):
    """
    Create preprocessing pipeline:
    - scale numerical features
    - one-hot encode categorical features
    """
    preprocessor = ColumnTransformer([
        (
            'num',
            StandardScaler(),
            numeric_cols
        ),
        (
            'cat',
            OneHotEncoder(
                drop='first',
                handle_unknown='ignore'
            ),
            CATEGORICAL_COLS
        )
    ])

    return preprocessor

def test_train_split(X,y):
    """
    Split data into:
    - train
    - validation
    - test
    """
    # First split:
    # 60% train
    # 40% temporary
    X_train, X1_test, y_train, y1_test = train_test_split(
        X,
        y,
        test_size=0.4,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Second split:
    # 20% validation
    # 20% test
    X_val, X_test, y_val, y_test = train_test_split(
        X1_test,
        y1_test,
        test_size=0.5,
        stratify=y1_test,
        random_state=RANDOM_STATE
    )

    # Print dataset sizes
    print("\nDataset Sizes")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    return X_train, y_train, X_val, X_test, y_val, y_test

def cap_outliers(X_train, X_val, X_test, cols):
    """
    IQR-based capping fitted on train, applied to all splits.
    """

    # deciding to not capp the CREDIT_UTILIZATION because it vary a lot but still is very meaningful
    EXCLUDE = {"PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "PAYMENT_RATIO", "MONTHS_DELAYED", "CREDIT_UTILIZATION"}
    COLS_TO_CAP = [c for c in cols if c not in EXCLUDE]

    bounds = {}
    for col in COLS_TO_CAP:
        if col not in X_train.columns:
            continue
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        bounds[col] = (lower, upper)

    for df in [X_train, X_val, X_test]:
        for col, (lower, upper) in bounds.items():
            df[col] = df[col].clip(lower, upper)

    return X_train, X_val, X_test, bounds

def run():
    """
    Complete preprocessing pipeline.
    """
     # Load and clean dataset
    X, y, numeric_cols = load_prepare_data()
    
    # Split dataset
    X_train, y_train,  X_val, X_test, y_val, y_test = test_train_split(X,y)
    X_train, X_val, X_test, _ = cap_outliers(X_train, X_val, X_test, numeric_cols)

    # Create preprocessor
    preprocessor = create_preprocessor(numeric_cols)

    return X_train, y_train, X_val, X_test, y_val, y_test, preprocessor

if __name__ == "__main__":
    run()