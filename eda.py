import os
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import CATEGORICAL_COLS, NUMERIC_COLS
from ucimlrepo import fetch_ucirepo
import pandas as pd

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



def run():
    # get dataset from UCI and rename columns 
    dataset = fetch_ucirepo(id=350)
    X = dataset.data.features
    y = dataset.data.targets
    df = pd.concat([X, y], axis=1)
    df = df.rename(columns=COL_NAMES)

    # Shape, Info, Describtion
    print("\nShape:", df.shape)
    print("\nColumn info:")
    df.info()
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe())

    # Class distribution
    counts = df["Y"].value_counts()
    percentages = df["Y"].value_counts(normalize=True) * 100
    print("\nClass counts:\n", counts)
    print("\nClass percentages:\n", percentages)
    ratio = counts.max() / counts.min()
    print(f"\nImbalance ratio: {ratio:.2f} : 1")

    # Show values
    print("\nNumber of different values: ", df.nunique().sort_values)


    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x="Y", data=df)
    plt.title("Class distribution: default payment next month")
    plt.xlabel("Default (0 = no, 1 = yes)")
    plt.ylabel("Count")
    for i, count in enumerate(counts):
        ax.text(i, count, str(count), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig("phases/results/EDA/ratio.png")
    plt.close()

    # Numeric feature histogram
    df[NUMERIC_COLS].hist(figsize=(15, 12), bins=30)
    plt.suptitle("Histograms of numeric features", y=1.02)
    plt.tight_layout()
    plt.savefig("phases/results/EDA/histogram_numerical_features.png")
    plt.close()

    # Numeric features boxplots 
    n, cols = len(NUMERIC_COLS), 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    for ax, col in zip(axes.flat, NUMERIC_COLS):
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(col)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    plt.suptitle("Boxplots of numeric features", y=1.02)
    plt.tight_layout()
    plt.savefig("phases/results/EDA/distribultion_numerical_features.png")
    plt.close()

    # Categorical features value counts & distributions
    for col in CATEGORICAL_COLS:
        print(f"\n{col} value counts:")
        print(df[col].value_counts().sort_index())

    n, cols = len(CATEGORICAL_COLS), 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    for ax, col in zip(axes.flat, CATEGORICAL_COLS):
        sns.countplot(x=col, data=df, ax=ax)
        ax.set_title(col)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    plt.suptitle("Distributions of categorical features", y=1.02)
    plt.tight_layout()
    plt.savefig("phases/results/EDA/distribultion_categorical_features.png")
    plt.close()

    # Numeric features vs target 
    n, cols = len(NUMERIC_COLS), 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    for ax, col in zip(axes.flat, NUMERIC_COLS):
        sns.boxplot(x="Y", y=col, data=df, ax=ax)
        ax.set_title(col)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    plt.suptitle("Numeric features by default status", y=1.02)
    plt.tight_layout()
    plt.savefig("phases/results/EDA/distribultion_features_vs_target.png")
    plt.close()

    # Default rate per categorical category  
    print("\nDefault rate by category:")
    for col in CATEGORICAL_COLS:
        rates = df.groupby(col)["Y"].mean().round(3) * 100
        print(f"\n{col}:\n{rates}")

    n, cols = len(CATEGORICAL_COLS), 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    for ax, col in zip(axes.flat, CATEGORICAL_COLS):
        rates = df.groupby(col)["Y"].mean() * 100
        rates.plot(kind="bar", ax=ax)
        ax.set_title(f"Default rate by {col}")
        ax.set_ylabel("Default rate (%)")
        ax.axhline(df["Y"].mean() * 100, color="red", linestyle="--", label="Overall rate")
        ax.legend()
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig("phases/results/EDA/defaultrate_per_category.png")
    plt.close()

    # Correlation matrix  
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8})
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig("phases/results/EDA/heatmap.png")
    plt.close()

    print("\nCorrelation with target (Y), sorted:")
    print(corr["Y"].drop("Y").sort_values(key=abs, ascending=False).round(3))

    # Missing values & duplicates  
    print("\nMissing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "None")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    print("\nEDA finished")


if __name__ == "__main__":
    run()
