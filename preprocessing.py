from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import StandardScaler
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
CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE",
                    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

NUMERIC_COLS = ["LIMIT_BAL", "AGE",
                "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
                "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]


def preprocessing(df):
    df = df.copy()

    # Dataset does not have missing values

    # Education (1 = graduate school; 2 = university; 3 = high school; 4 = others) but the dataset contains values with cathegory 0,5,6 so merge into others
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    # Marital status (1 = married; 2 = single; 3 = others) but the dataset contains values with cathegory 0 so merge into others
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    # SEX as binary (male=0, female=1) because now 1,2
    df["SEX"] = (df["SEX"] == 2).astype(int)

    # One-hot encoding: create binary columns for EDUCATION and MARRIAGE to get a better understanding of e.g. does being a university graduate increase default risk compared to a graduate school graduate?
    # drop_first=True -> avoid multicollinearity (one can only be in one category, which si the highest grade)
    edu_dummies = pd.get_dummies(df["EDUCATION"], prefix="EDU", drop_first=True)
    mar_dummies = pd.get_dummies(df["MARRIAGE"], prefix="MAR", drop_first=True)
    df = df.drop(columns=["EDUCATION", "MARRIAGE"])
    df = pd.concat([df, edu_dummies, mar_dummies], axis=1)

    # Feature engineering
    # How much of their credit limit they're using — high utilization is a classic default risk signal
    df['CREDIT_UTILIZATION'] = df['BILL_AMT1'] / df['LIMIT_BAL']
    # Did they pay the full bill, half, or almost nothing? Clip to [0,1]: >1 means overpaid (treat as 1)
    # Replace 0 bill amounts with NaN first to avoid division by zero, then fill with 0 (no bill = no ratio)
    df['PAYMENT_RATIO'] = (df['PAY_AMT1'] / df['BILL_AMT1'].replace(0, float('nan'))).clip(0, 1).fillna(0)
    # How many of the 6 months had a payment delay (PAY > 0 means delayed)
    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    df['MONTHS_DELAYED'] = (df[pay_cols] > 0).sum(axis=1)

    # get Features and target seperately
    feature_names = [c for c in df.columns if c != "Y"]
    X = df[feature_names].values.astype(float)
    y = df["Y"].values.astype(int)

    # Train/test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\nPreprocessing complete.")
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Train default rate: {y_train.mean():.3f}  |  Test default rate: {y_test.mean():.3f}")
    print(f"  Class imbalance ratio (train): {(y_train == 0).sum() / (y_train == 1).sum():.2f} : 1")

    return X_train, X_test, y_train, y_test, feature_names, scaler


def run():
    # get dataset from UCI and rename columns 
    dataset = fetch_ucirepo(id=350)
    X = dataset.data.features
    y = dataset.data.targets
    df = pd.concat([X, y], axis=1)
    df = df.rename(columns=COL_NAMES)

    X_train, X_test, y_train, y_test, feature_names, scaler = preprocessing(df)
    return X_train, X_test, y_train, y_test, feature_names, scaler


if __name__ == "__main__":
    run()
