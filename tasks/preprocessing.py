from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Exploratory Data Analysis (EDA)
def EDA(df, X, y, categorical_cols, numeric_cols):
    # Dataset overview
    print("\nShape: ", df.shape)
    print("\nColumn info:")
    df.info()
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe())

    # Counts and percentages
    counts = df["Y"].value_counts()
    percentages = df["Y"].value_counts(normalize=True) * 100

    print("\nClass counts:")
    print(counts)
    print("\nClass percentages:")
    print(percentages)

    # Ratio
    ratio = counts.max() / counts.min()
    print("\nImbalance ratio: {:.2f} : 1".format(ratio))

    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x="Y", data=df)
    plt.title("Class distribution: default payment next month")
    plt.xlabel("Default (0 = no, 1 = yes)")
    plt.ylabel("Count")
    for i, count in enumerate(counts):
        ax.text(i, count, str(count), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(f"results/EDA/ratio.png")
    plt.close()

    # Distribution of numeric features
    df[numeric_cols].hist(figsize=(15, 12), bins=30)
    plt.suptitle("Histograms of numeric features", y=1.02)
    plt.tight_layout()
    plt.savefig(f"results/EDA/histogram_numerical_features.png")
    plt.close()

    n = len(numeric_cols)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    for ax, col in zip(axes.flat, numeric_cols):
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(col)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    plt.suptitle("Boxplots of numeric features", y=1.02)
    plt.tight_layout()
    plt.savefig(f"results/EDA/distribultion_numerical_features.png")
    plt.close()


    # Distribution of categorical features
    for col in categorical_cols:
        print(f"\n{col} value counts:")
        print(df[col].value_counts().sort_index())

    n = len(categorical_cols)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    for ax, col in zip(axes.flat, categorical_cols):
        sns.countplot(x=col, data=df, ax=ax)
        ax.set_title(col)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    plt.suptitle("Distributions of categorical features", y=1.02)
    plt.tight_layout()
    plt.savefig(f"results/EDA/distribultion_categorical_features.png")
    plt.close()

    # features vs target
    n = len(numeric_cols)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    for ax, col in zip(axes.flat, numeric_cols):
        sns.boxplot(x="Y", y=col, data=df, ax=ax)
        ax.set_title(col)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    plt.suptitle("Numeric features by default status", y=1.02)
    plt.tight_layout()
    plt.savefig(f"results/EDA/distribultion_features_vs_target.png")
    plt.close()


    # Categorical features: default rate per category
    print("\nDefault rate by category:")
    for col in categorical_cols:
        rates = df.groupby(col)["Y"].mean().round(3) * 100
        print(f"\n{col}:")
        print(rates)

    n = len(categorical_cols)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    for ax, col in zip(axes.flat, categorical_cols):
        rates = df.groupby(col)["Y"].mean() * 100
        rates.plot(kind="bar", ax=ax)
        ax.set_title(f"Default rate by {col}")
        ax.set_ylabel("Default rate (%)")
        ax.axhline(df["Y"].mean() * 100, color="red", linestyle="--",
                   label="Overall rate")
        ax.legend()
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(f"results/EDA/defaultrate_per_category.png")
    plt.close()


    # Correlation matrix -> MAYBE NOT NESSASARY
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8})
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(f"results/EDA/heatmap.png")
    plt.close()

    print(f"\nCorrelation with target ({y}), sorted:")
    print(corr["Y"].drop("Y").sort_values(key=abs, ascending=False).round(3))

    # Missing values
    print("\nMissing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "None")

    # Duplicates
    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    print("EDA finished")

# Data preprocessing (missing values, encoding, scaling)
def preprocessing(df, X, y):
    # Encoding?
    print("Start Preprocessing")


def run():
    # fetch dataset 
    default_of_credit_card_clients = fetch_ucirepo(id=350) 
    
    # data (as pandas dataframes) 
    X = default_of_credit_card_clients.data.features 
    y = default_of_credit_card_clients.data.targets 
    
    # metadata 
    print("\nMetadata: ", default_of_credit_card_clients.metadata) 
    # variable information 
    print("\nVariables: ", default_of_credit_card_clients.variables) 

    df = pd.concat([X, y], axis=1)

    # ucimlrepo returns features as X1..X23; rename to descriptive names
    col_names = {
        "X1": "LIMIT_BAL", "X2": "SEX", "X3": "EDUCATION", "X4": "MARRIAGE",
        "X5": "AGE", "X6": "PAY_0", "X7": "PAY_2", "X8": "PAY_3",
        "X9": "PAY_4", "X10": "PAY_5", "X11": "PAY_6",
        "X12": "BILL_AMT1", "X13": "BILL_AMT2", "X14": "BILL_AMT3",
        "X15": "BILL_AMT4", "X16": "BILL_AMT5", "X17": "BILL_AMT6",
        "X18": "PAY_AMT1", "X19": "PAY_AMT2", "X20": "PAY_AMT3",
        "X21": "PAY_AMT4", "X22": "PAY_AMT5", "X23": "PAY_AMT6",
    }
    df = df.rename(columns=col_names)

    # numerical and categorical column
    categorical_cols = ["SEX", "EDUCATION", "MARRIAGE",
                        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    
    numeric_cols = ["LIMIT_BAL", "AGE",
                    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
                    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
    
    EDA(df, X, y, categorical_cols, numeric_cols)
    preprocessing(df, X, y)
    

if __name__ == "__main__":
    run()