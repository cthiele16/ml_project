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

def train_model(X_train, y_train, preprocessor):

    pipeline = Pipeline([
        ('preprocessor', preprocessor),

        ('model', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ))
    ])

    param_grid = {
        'model__C': [0.01, 0.1, 1, 10, 100]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

    return grid.best_estimator_

def coefficient_analysis(best_model):
    coef = best_model.named_steps['model'].coef_[0]

    feature_names = best_model.named_steps[
        'preprocessor'].get_feature_names_out()

    coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coef
    })

    coef_sorted = coef_df.sort_values(
        by='coefficient',
        key=abs,
        ascending=False
    )

    return coef_sorted

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

def find_best_threshold(y_true, y_prob):
    df = evaluate_thresholds(y_true, y_prob)

    best_row = df.loc[df["f1"].idxmax()]

    return best_row["threshold"]


def run():
    X_train, y_train, X_val, X_test, y_val, y_test, preprocessor= run_preprocessing()
    
    # Train Model
    best_model = train_model(
        X_train,
        y_train,
        preprocessor
    )

    # Coefficient analysis
    coef_analysis = coefficient_analysis(best_model)
    print(coef_analysis)

    # Validation -> Threshold tuning
    y_val_prob = best_model.predict_proba(X_val)[:, 1]

    best_threshold = find_best_threshold(
        y_val,
        y_val_prob
    )

    print("\nBest threshold:", best_threshold)

    # Final test evaluation
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

    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))


    return best_model, best_threshold

if __name__ == "__main__":
    run()