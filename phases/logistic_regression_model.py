import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    average_precision_score
)

from preprocessing import RANDOM_STATE, run as run_preprocessing
from utils import save_results


def train_model(X_train, y_train, preprocessor):
    """
        Train Logistic Regression model using:
        - preprocessing pipeline
        - cross-validation
        - hyperparameter tuning
    """

    #Build pipeline for preprocessing 
    pipeline = Pipeline([
        ('preprocessor', preprocessor),

        ('model', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ))
    ])

    # Regularization strengths to test different values of C
    param_grid = {
        'model__C': [0.01, 0.1, 1, 10, 100]
    }

    # Stratified CV
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # Grid search to test C-values using CV
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    #Train all models
    grid.fit(X_train, y_train)

    # Print best hyperparameter & best average CV score
    print("Best params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

    #return best trained pipeline
    return grid.best_estimator_

def coefficient_analysis(best_model):
    """
    Extract and analyze logistic regression coefficients.
    """

    #Extract model coefficients
    coef = best_model.named_steps['model'].coef_[0]
    # Extract feature names
    feature_names = best_model.named_steps[
        'preprocessor'].get_feature_names_out()

    #Create dataframe with features and coefficients
    coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coef
    })

    # Sort by absolute coefficient size
    # Largest effect first
    coef_sorted = coef_df.sort_values(
        by='coefficient',
        key=abs,
        ascending=False
    )

    #Return sorted coefficients
    return coef_sorted

def evaluate_thresholds(y_true, y_prob):
    """
    Evaluate classification thresholds
    and calculate metrics for each threshold.
    """
    # Thresholds to test
    thresholds = np.arange(0.1, 0.9, 0.05)

    results = []

    # Test all thresholds
    for t in thresholds:
         
         # Convert probabilities to binary predictions
        preds = (y_prob >= t).astype(int)
        
        # Store evaluation metrics
        results.append({
            'threshold': t,
            'f1': f1_score(y_true, preds),
            'precision': precision_score(y_true, preds),
            'recall': recall_score(y_true, preds),
            'accuracy': accuracy_score(y_true, preds)
        })
    
    # Return all threshold results
    return pd.DataFrame(results)

def find_best_threshold(y_true, y_prob):
    """
    Select threshold with highest F1-score.
    """

    # Evaluate all thresholds
    df = evaluate_thresholds(y_true, y_prob)
    
    # Find row with highest F1-score
    best_row = df.loc[df["f1"].idxmax()]
     
     # Return best threshold
    return best_row["threshold"]


def run():
    # Run preprocessing
    X_train, y_train, X_val, X_test, y_val, y_test, preprocessor= run_preprocessing()
    
    # Train Model using CV
    best_model = train_model(
        X_train,
        y_train,
        preprocessor
    )

    # Analyze feature coefficients
    coef_analysis = coefficient_analysis(best_model)
    print("\nCoefficient Analysis")
    print(coef_analysis)

    # Predict probabilities for validation data
    y_val_prob = best_model.predict_proba(X_val)[:, 1]

    #Find threshold with best F1-score
    best_threshold = find_best_threshold(
        y_val,
        y_val_prob
    )

    print("\nBest threshold:", best_threshold)

    # Final evaluation on test set using best threshold
    y_test_prob = best_model.predict_proba(X_test)[:, 1]

    y_test_pred = (y_test_prob >= best_threshold).astype(int)

    print("\nFINAL TEST RESULTS")
    print("-" * 50)

    print("\nAUC-ROC:")
    print(roc_auc_score(y_test, y_test_prob))

    print("\nAUPR:")
    print(
        average_precision_score(
            y_test,
            y_test_prob
        )
    )

    #Full classification metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

    save_results("logisticRegression", y_test, y_test_pred, y_test_prob,
             best_params={"C": best_model.named_steps["model"].C})

    #return best model and best threshold
    return best_model, best_threshold

if __name__ == "__main__":
    run()