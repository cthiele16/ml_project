import json
import os

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

def save_results(model_name, y_test, y_test_pred, y_test_score,
                 best_params=None, variants=None):
    
    """
    Save classification metrics to results/<model_name>.json

    Parameters
    model_name : Used as filename
    y_test : True test labels
    y_test_pred : Predicted binary labels (from .predict())
    y_test_score : Continuous scores for ranking metrics.
        - Logistic Regression / Random Forest / Boosting: .predict_proba(X_test)[:, 1]
        - SVM: .decision_function(X_test)
    best_params : Hyperparameters of the chosen final model.
    variants : list of dict, optional
        - Per-variant results (e.g. one row per SVM kernel).
        - Useful when several variants of the same model were compared.
    """

    metrics = {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "f1": f1_score(y_test, y_test_pred),
        "roc_auc": roc_auc_score(y_test, y_test_score),
        "aupr": average_precision_score(y_test, y_test_score),
    }

    # Save best hyperparameters if given
    if best_params is not None:
        clean_params = {}
        for key, value in best_params.items():
            clean_params[key] = str(value)  # convert to string so JSON can save it
        metrics["best_params"] = clean_params

    # Save per-variant results if given (e.g. one entry per SVM kernel)
    if variants is not None:
        clean_variants = []
        for variant in variants:
            variant_copy = dict(variant)
            # Also clean the best_params inside each variant
            if "best_params" in variant_copy:
                clean_inner = {}
                for key, value in variant_copy["best_params"].items():
                    clean_inner[key] = str(value)
                variant_copy["best_params"] = clean_inner
            clean_variants.append(variant_copy)
        metrics["variants"] = clean_variants

    path = f"phases/results/{model_name}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved results in JSON format")