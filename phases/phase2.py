import pandas as pd
import numpy as np
from preprocessing import run as preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report
)

def find_best_model(X_train, y_train, X_test, y_test, feature_names):
    
    print("Find best model")

    C_values = [0.01, 0.1, 1, 10, 100]

    coefs = []
    results = []

    best_c = None
    best_model = None
    best_f1 = -1

    for C in C_values:
        model = LogisticRegression(
            C=C,
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )

        model.fit(X_train, y_train)
        coefs.append(model.coef_[0])

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        f1 = f1_score(y_test, y_pred)

        results.append({
            "C": C,
            "f1": f1,
            "auc": roc_auc_score(y_test, y_prob)
})

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_c = C
    
    coef_df = pd.DataFrame(
        coefs, 
        columns=feature_names
    )
    
    coef_df["C"] = C_values

    print("\nCoefficient Table:")
    print(coef_df)

    return best_model, pd.DataFrame(results), best_c

def evaluate_thresholds(y_true, y_prob):
    thresholds = np.arange(0.1, 0.9, 0.05)

    results = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)

        results.append({
            'threshold': t,
            'f1': f1_score(y_true, preds),
            'precision': precision_score(y_true, preds),
            'recall': recall_score(y_true, preds),
            'accuracy': accuracy_score(y_true, preds)
        })

    return pd.DataFrame(results)

def find_best_threshold(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    
    threshold_results = evaluate_thresholds(y_test, y_prob)
    
    best_row = threshold_results.loc[
        threshold_results['f1'].idxmax()
        ]

    best_threshold = best_row['threshold']

    return best_threshold, y_prob, threshold_results


def run():

    # STEP 1: preprocessing     
    X_train, X_test, y_train, y_test, feature_names = preprocess()

    # STEP 2: model selection (C)
    best_model, results_df, best_c = find_best_model(
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        feature_names
    )

    print(results_df)
    print("Best C:", best_c)

    # STEP 3: threshold tuning
    best_threshold, y_prob, threshold_results = find_best_threshold(
        best_model, X_test, y_test
    )

    # STEP 4: final prediction
    y_pred = (y_prob >= best_threshold).astype(int)

    # STEP 5: evaluation
    print("Best threshold results:", best_threshold)
    print("AUC-ROC Score:", roc_auc_score(y_test, y_prob))
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    run()