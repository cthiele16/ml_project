import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
)

from preprocessing import RANDOM_STATE, run as run_preprocessing
from utils import save_results


def train_model(X_train, y_train, preprocessor):
    # Train SVM models with different kernels using: preprocessing pipeline, subsampled grid search (5000 points), refit best model on full training set

    # Subsample for grid search: full samples + multiple kernels + CV folds is too slow
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(X_train.shape[0], size=5000, replace=False)
    X_sub = X_train.iloc[idx]
    y_sub = y_train.iloc[idx]

    # C: trade-off between margin width and misclassification
        # Small C (0.01): wide margin, may underfit
        # Large C (10):   narrow margin, may overfit
    # class_weight="balanced": compensates for class imbalance
    # scoring="roc_auc": evaluates ranking of defaulters across all thresholds
    grids = {
        # total fits = candidates × CV folds (5)
        
        # linear 
        "Linear": GridSearchCV(
            Pipeline([
                ('preprocessor', preprocessor),
                ('model', SVC(kernel="linear", class_weight="balanced")),
            ]),
            param_grid={"model__C": [0.01, 0.1, 1, 10]},
            cv=5, scoring="roc_auc", n_jobs=-1, verbose=1,
        ),
        # degree=2 -> quadratic boundary, degree=3 -> cubic
        "Poly": GridSearchCV(
            Pipeline([
                ('preprocessor', preprocessor),
                ('model', SVC(kernel="poly", class_weight="balanced")),
            ]),
            param_grid={"model__C": [0.1, 1, 10], "model__degree": [2, 3]},
            cv=5, scoring="roc_auc", n_jobs=-1, verbose=1,
        ),
        # Small gamma (0.01): smooth, far-reaching influence
        # Large gamma: wiggly boundary that hugs training points
        "RBF": GridSearchCV(
            Pipeline([
                ('preprocessor', preprocessor),
                ('model', SVC(kernel="rbf", class_weight="balanced")),
            ]),
            param_grid={"model__C": [0.1, 1, 10], "model__gamma": ["scale", "auto", 0.01]},
            cv=5, scoring="roc_auc", n_jobs=-1, verbose=1,
        ),
    }

    results = []

    for name, grid in grids.items():
        print(f"\nTraining SVM with {name} kernel")

        # Grid search on subsample
        grid.fit(X_sub, y_sub)

        # Refit best pipeline on full training set
        best = grid.best_estimator_
        best.fit(X_train, y_train)

        print("Best params: ", grid.best_params_)
        print(f"CV ROC-AUC (on subsample): {grid.best_score_:.4f}")

        results.append((name, grid, best))

    return results


def run():
    X_train, y_train, X_val, X_test, y_val, y_test, preprocessor = run_preprocessing()

    results_list = train_model(X_train, y_train, preprocessor)

    summary_rows = []

    for name, grid, best in results_list:
        y_val_score = best.decision_function(X_val)
        y_val_pred = best.predict(X_val)
        
        summary_rows.append({
            "kernel": name,
            "best_params": grid.best_params_,
            "cv_roc_auc": grid.best_score_,
            "val_roc_auc": roc_auc_score(y_val, y_val_score),
            "val_aupr": average_precision_score(y_val, y_val_score),
            "val_f1": f1_score(y_val, y_val_pred),
        })

    summary = pd.DataFrame(summary_rows).sort_values("val_roc_auc", ascending=False)
    print(summary.to_string(index=False))

    # Then only evaluate the winner on test
    best_name = summary.iloc[0]["kernel"]

    models_by_name = {}
    for name, grid, best in results_list:
        models_by_name[name] = best

    best_model = models_by_name[best_name]

    y_test_score = best_model.decision_function(X_test)
    y_test_pred = best_model.predict(X_test)
    print(f"\nFinal test results for {best_name} trained on full training data:")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_test_score):.4f}")
    print(f"AUPR:    {average_precision_score(y_test, y_test_score):.4f}")
    print(classification_report(y_test, y_test_pred))

    save_results("svm",y_test, y_test_pred, y_test_score,best_params=summary.iloc[0]["best_params"], variants=summary.to_dict(orient="records"))
    return summary 


if __name__ == "__main__":
    run()
