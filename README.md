# Credit Default Prediction Project

This project implements a comprehensive machine learning pipeline to predict the likelihood of credit card default using the UCI "Default of Credit Card Clients" dataset. 

## Prerequisites

The project requires Python 3.x and the following libraries:
* `ucimlrepo`: To fetch the dataset directly from UCI.
* `scikit-learn`: For modeling and preprocessing.
* `pandas` & `numpy`: For data manipulation.
* `matplotlib` & `seaborn`: For visualizations.
* `optuna`: For hyperparameter optimization in ensemble methods.

## Installation

Install all required dependencies via pip:

```bash
pip install ucimlrepo scikit-learn pandas numpy matplotlib seaborn optuna
```

## How to Run

All project phases are executed through the central entry point `main.py`. Commands should be executed from the `ml_project/` directory.

```bash
python main.py <phasename>
```

### Available Phases:

1.  **`eda`**: Runs Exploratory Data Analysis, printing statistics and saving plots to `phases/results/EDA/`.
2.  **`preprocessing`**: Handles data cleaning, feature engineering (e.g., Credit Utilization), and creates the train/val/test splits.
3.  **`unsupervised`**: Performs PCA for dimensionality analysis and clustering (K-Means/GMM) to inspect data structure.
4.  **`baseline`**: Establishes a performance floor using a random `DummyClassifier`.
5.  **`logistic_regression`**: Fits a regularized Logistic Regression with automated threshold tuning for F1-score.
6.  **`svm`**: Executes a grid search across Linear, Poly, and RBF kernels.
7.  **`random_forest`**: Performs hyperparameter tuning using Optuna for Random Forest, AdaBoost, and Gradient Boosting.
8.  **`modelcomparison`**: Aggregates results from all models. If a model's results are missing, it will automatically run that model's training phase first.

## Project Structure & Requirements

The project satisfies the following requirements:
* **Structural Analysis**: PCA and Clustering.
* **Model Diversity**: Baseline, Linear (Logistic), Kernel (SVM), and Ensembles (RF/Boosting).
* **Validation**: Proper Train/Validation/Test splitting and Cross-Validation.
* **Metrics**: Evaluation based on Accuracy, F1-score, ROC-AUC, and AUPR.

The final performance comparison is output as a table sorted by **ROC-AUC** to identify the most robust model for the dataset.
