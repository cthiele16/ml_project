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
    """
    Train Logistic Regression models with different regularization strengths (C),
    compare their performance, and return the best model.
    """
    print("Find best model")

    # Different regularization strengths to test
    C_values = [0.01, 0.1, 1, 10, 100]

    # Store coefficients for later analysis
    coefs = []
    # Store evaluation results for each C
    results = []

     # Variables used to keep track of best model
    best_c = None
    best_model = None
    best_f1 = -1

    # Train one model for each C value
    for C in C_values:

        # Create Logistic Regression model
        model = LogisticRegression(
            C=C,
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )

         # Train model on training data
        model.fit(X_train, y_train)
        # Save model coefficients, model.coef_[0] contains one coefficient for each feature
        coefs.append(model.coef_[0])

         # Predict probabilities for class 1 (default)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Convert probabilities into binary predictions using threshold = 0.5
        y_pred = (y_prob >= 0.5).astype(int)

        # Compute F1-score
        f1 = f1_score(y_test, y_pred)
        
        # Save model performance
        results.append({
            "C": C,
            "f1": f1,
            "auc": roc_auc_score(y_test, y_prob)
        })

        # Update best model if current model performs better
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_c = C
    
    # Create dataframe of coefficients
    # Each row = one C value
    # Each column = one feature coefficient
    coef_df = pd.DataFrame(
        coefs, 
        columns=feature_names
    )
    
    # Add C column for easier comparison
    coef_df["C"] = C_values

    print("\nCoefficient Table:")
    print(coef_df)

    # Return:
    # - best trained model
    # - results dataframe
    # - best regularization parameter
    return best_model, pd.DataFrame(results), best_c

def evaluate_thresholds(y_true, y_prob):
    """
    Evaluate several classification thresholds
    and compute metrics for each threshold.
    """

    # Thresholds to test
    thresholds = np.arange(0.1, 0.9, 0.05)

    results = []

    # Evaluate every threshold
    for t in thresholds:
         #Convert probabilities to predictions
        preds = (y_prob >= t).astype(int)

        # Save evaluation metrics
        results.append({
            'threshold': t,
            'f1': f1_score(y_true, preds),
            'precision': precision_score(y_true, preds),
            'recall': recall_score(y_true, preds),
            'accuracy': accuracy_score(y_true, preds)
        })

    # Return dataframe with all threshold results
    return pd.DataFrame(results)

def find_best_threshold(model, X_test, y_test):
    """
    Find the threshold that gives the highest F1-score.
    """

    # Predict probabilities for positive class
    y_prob = model.predict_proba(X_test)[:, 1]
    # Evaluate many thresholds
    threshold_results = evaluate_thresholds(y_test, y_prob)
    
    # Select row with highest F1-score
    best_row = threshold_results.loc[
        threshold_results['f1'].idxmax()
        ]

    # Extract best threshold value
    best_threshold = best_row['threshold']

    return best_threshold, y_prob, threshold_results


def run():

    # STEP 1: Preprocessing     
    X_train, X_test, y_train, y_test, feature_names = preprocess()

    # STEP 2: Model selection (regularizaton)
    best_model, results_df, best_c = find_best_model(
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        feature_names
    )

    print(results_df)
    print("Best C:", best_c)

    # STEP 3: Threshold Tuning
    best_threshold, y_prob, threshold_results = find_best_threshold(
        best_model, X_test, y_test
    )

    # STEP 4: Final Prediction
    y_pred = (y_prob >= best_threshold).astype(int)

    # STEP 5: Evaluation
    print("Best threshold results:", best_threshold)
    print("AUC-ROC Score:", roc_auc_score(y_test, y_prob))
    print(classification_report(y_test, y_pred))

# Run full pipeline:
if __name__ == "__main__":
    run()