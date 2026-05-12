from sklearn.compose import ColumnTransformer
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

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

NUMERIC_COLS = ["LIMIT_BAL", "AGE",
                "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
                "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
                "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"
                ]

def load_prepare_data():
    dataset = fetch_ucirepo(id=350)

    X = dataset.data.features
    y = dataset.data.targets

    df = pd.concat([X, y], axis=1)

    df = df.rename(columns=COL_NAMES)

    # CLEAN
    df["EDUCATION"] = df["EDUCATION"].replace({
        0: 4,
        5: 4,
        6: 4
    })

    df["MARRIAGE"] = df["MARRIAGE"].replace({
        0: 3
    })

    # female = 1, male = 0
    df["SEX"] = (df["SEX"] == 2).astype(int)

    #FEATURE ENGINEERING
    df["CREDIT_UTILIZATION"] = (
        df["BILL_AMT1"] / df["LIMIT_BAL"]
    )

    df["PAYMENT_RATIO"] = (
        df["PAY_AMT1"]
        / df["BILL_AMT1"].replace(0, np.nan)
    ).clip(0, 1).fillna(0)

    pay_cols = [
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6"
    ]

    df["MONTHS_DELAYED"] = (
        df[pay_cols] > 0
    ).sum(axis=1)

    engineered_cols = [
        "CREDIT_UTILIZATION",
        "PAYMENT_RATIO",
        "MONTHS_DELAYED"
    ]

    numeric_cols = NUMERIC_COLS + engineered_cols

    #FEATURES + TARGET
    X = df.drop(columns=["Y"])
    y = df["Y"]

    return X, y, numeric_cols

def create_preprocessor(numeric_cols):
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
    # Split Data

    X_train, X1_test, y_train, y1_test = train_test_split(
        X,
        y,
        test_size=0.4,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X1_test,
        y1_test,
        test_size=0.5,
        stratify=y1_test,
        random_state=RANDOM_STATE
    )

    print("\nDataset Sizes")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    return X_train, y_train, X_val, X_test, y_val, y_test

def run():
    X, y, numeric_cols = load_prepare_data()
    
    X_train, y_train,  X_val, X_test, y_val, y_test =test_train_split(X,y)

    preprocessor = create_preprocessor(numeric_cols)

    return X_train, y_train, X_val, X_test, y_val, y_test, preprocessor


if __name__ == "__main__":
    run()