import optuna
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier, 
    AdaBoostClassifier, 
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from preprocessing import RANDOM_STATE, run as run_preprocessing
from utils import save_results
import optuna.visualization as vis
import time

def random_forest(trial, X_train, y_train, X_val, y_val):
    # Define search space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        #Handle imbalanced dataset
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
        "max_features": trial.suggest_categorical(
            "max_features",
            ["sqrt", "log2"]
        ),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    # return model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, val_probs)

def random_forest_oob(trial, X_train_large, y_train_large):
    # Define search space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "oob_score": True,  # CRITICAL: Must be True to calculate OOB score
    }

    model = RandomForestClassifier(**params)
    model.fit(X_train_large, y_train_large)
    return model.oob_score_

def gbm_objective(trial, X_train_large, y_train_large):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
        "random_state": RANDOM_STATE
    }
    
    model = GradientBoostingClassifier(**params)
    
    # 3-Fold CV
    scores = cross_val_score(model, X_train_large, y_train_large, cv=3, n_jobs=-1, scoring="roc_auc")
    return np.mean(scores)

def adaboost_objective(trial, X_train_large, y_train_large):
    # Tune the base weak learner along with AdaBoost parameters
    max_depth = trial.suggest_int("max_depth", 1, 5)
    class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
    
    params = {
        "estimator": DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE, class_weight=class_weight),
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
        "random_state": RANDOM_STATE
    }
    
    model = AdaBoostClassifier(**params)
    
    # 3-Fold CV to evaluate parameters on the large training pool
    scores = cross_val_score(model, X_train_large, y_train_large, cv=3, n_jobs=-1, scoring="roc_auc")
    return np.mean(scores)

def run():
    X_train, y_train, X_val, X_test, y_val, y_test, _ = run_preprocessing()
    num_trials = 30 #can set up to 100, needed to decrease it to run it (before it was 50)

    results = []
    feature_importances = {}

    # Store per-model test predictions, scores and best params so the winning
    # model can be saved with save_results (like in svm.py)
    model_preds = {}
    model_probs = {}
    model_params = {}

    # ------------------------------------------
    # MODEL 1: Random Forest (CV Tuning)
    # ------------------------------------------
    # Define model and execute hyperparameter search
    rf_study = optuna.create_study(direction="maximize")
    rf_study.optimize(lambda trial: random_forest(trial, X_train, y_train, X_val, y_val), n_trials=num_trials)
    
    # Best model found
    print(f"Random Forest Best Validation ROC-AUC: {rf_study.best_value:.4f}")
    print(f"Best Params: {rf_study.best_params}")

    #Visualize tradeoffs in validation
    vis.plot_parallel_coordinate(rf_study).write_image("phases/results/RF/rf_cv_parallel.png")
    vis.plot_optimization_history(rf_study).write_image("phases/results/RF/rf_cv_optimization_history.png")
    vis.plot_contour(rf_study, params=["max_depth", "n_estimators"]).write_image("phases/results/RF/rf_cv_contour.png")
    
    # Final model assessment
    best_rand_model = RandomForestClassifier(**rf_study.best_params, random_state=RANDOM_STATE, n_jobs=-1)
    start = time.perf_counter()
    best_rand_model.fit(X_train, y_train)
    rf_val_time = time.perf_counter() - start

    test_preds = best_rand_model.predict(X_test)
    test_probs = best_rand_model.predict_proba(X_test)[:, 1]

    # Document results
    rf_val_acc = accuracy_score(y_test, test_preds)
    rf_val_f1 = f1_score(y_test, test_preds)
    rf_val_auc = roc_auc_score(y_test, test_probs)
    rf_val_precision = precision_score(y_test, test_preds)
    rf_val_recall = recall_score(y_test, test_preds)

    print(f"Final Test Accuracy: {rf_val_acc:.4f}")
    print(f"Final Test ROC-AUC: {rf_val_auc:.4f}")
    print("\nFinal Classification Report:\n", classification_report(y_test, test_preds))

    results.append({
        "Model": "RF Validation",
        "Accuracy": rf_val_acc,
        "F1": rf_val_f1,
        "Precision": rf_val_precision,
        "Recall": rf_val_recall,
        "ROC_AUC": rf_val_auc,
        "Train_Time_s": rf_val_time
    })

    feature_importances["RF Validation"] = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": best_rand_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    model_preds["RF Validation"] = test_preds
    model_probs["RF Validation"] = test_probs
    model_params["RF Validation"] = rf_study.best_params

    # ------------------------------------------
    # MODEL 2: Random Forest (OOB Tuning)
    # ------------------------------------------
    
    # Glue Train and Val together to make a larger training set, since we use OOB
    X_train_large = np.concatenate([X_train, X_val], axis=0)
    y_train_large = np.concatenate([y_train, y_val], axis=0)

    rf_oob_study = optuna.create_study(direction="maximize")
    rf_oob_study.optimize(
        lambda trial: random_forest_oob(trial, X_train_large, y_train_large), 
        n_trials=num_trials
    )

    print(f"Best OOB Score found during tuning: {rf_oob_study.best_value:.4f}")
    print(f"Best Params: {rf_oob_study.best_params}")

    # Visualize tradeoffs
    vis.plot_parallel_coordinate(rf_oob_study).write_image("phases/results/RF/rf_oob_parallel.png")
    vis.plot_optimization_history(rf_oob_study).write_image("phases/results/RF/rf_oob_optimization_history.png")
    
    # Final model assessment on test set
    best_oob_model = RandomForestClassifier(
        **rf_oob_study.best_params, 
        oob_score=True, 
        random_state=RANDOM_STATE, 
        n_jobs=-1
    )

    start = time.perf_counter()
    best_oob_model.fit(X_train_large, y_train_large)
    rf_oob_time = time.perf_counter() - start

    test_preds = best_oob_model.predict(X_test)
    test_probs = best_oob_model.predict_proba(X_test)[:, 1]

    #Calculate test result scores
    rf_oob_acc = accuracy_score(y_test, test_preds)
    rf_oob_f1 = f1_score(y_test, test_preds)
    rf_oob_precision = precision_score(y_test, test_preds)
    rf_oob_recall = recall_score(y_test, test_preds)
    rf_oob_auc = roc_auc_score(y_test, test_probs)

    print(f"Final OOB Score: {best_oob_model.oob_score_:.4f}")
    print(f"Final Test Accuracy: {rf_oob_acc:.4f}")
    print(f"Final Test ROC-AUC: {rf_oob_auc:.4f}")
    print("\nFinal Classification Report:\n", classification_report(y_test, test_preds))

    results.append({
        "Model": "RF OOB",
        "Accuracy": rf_oob_acc,
        "F1": rf_oob_f1,
        "Precision": rf_oob_precision,
        "Recall": rf_oob_recall,
        "ROC_AUC": rf_oob_auc,
        "Train_Time_s": rf_oob_time
    })

    feature_importances["RF OOB"] = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": best_oob_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    model_preds["RF OOB"] = test_preds
    model_probs["RF OOB"] = test_probs
    model_params["RF OOB"] = rf_oob_study.best_params

    # ------------------------------------------
    # MODEL 3: Random Forest (Gradient Boosting)
    # ------------------------------------------
    gd_study = optuna.create_study(direction="maximize")
    gd_study.optimize(lambda trial: gbm_objective(trial, X_train_large, y_train_large), n_trials=num_trials)

    # Visualize tradeoffs in Gradient Boosting validation
    vis.plot_parallel_coordinate(gd_study).write_image("phases/results/RF/rf_gb_parallel.png")
    vis.plot_optimization_history(gd_study).write_image("phases/results/RF/rf_gb_optimization_history.png")
    vis.plot_contour(gd_study, params=["max_depth", "n_estimators"]).write_image("phases/results/RF/rf_gb_contour.png")

    print(f"Best Gradient Boosting CV ROC-AUC: {gd_study.best_value:.4f}")
    print(f"Best Params: {gd_study.best_params}")
    
    best_grad_model = GradientBoostingClassifier(**gd_study.best_params, random_state=RANDOM_STATE)
    
    start = time.perf_counter()
    best_grad_model.fit(X_train_large, y_train_large)
    gb_time = time.perf_counter() - start

    test_preds = best_grad_model.predict(X_test)
    test_probs = best_grad_model.predict_proba(X_test)[:, 1]

    gb_acc = accuracy_score(y_test, test_preds)
    gb_f1 = f1_score(y_test, test_preds)
    gb_auc = roc_auc_score(y_test, test_probs)
    gb_precision = precision_score(y_test, test_preds)
    gb_recall = recall_score(y_test, test_preds)

    print(f"Final Gradient Boosting Test Accuracy: {gb_acc:.4f}")
    print(f"Final Gradient Boosting Test ROC-AUC: {gb_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, test_preds))

    results.append({
        "Model": "Gradient Boosting",
        "Accuracy": gb_acc,
        "F1": gb_f1,
        "Precision": gb_precision,
        "Recall": gb_recall,
        "ROC_AUC": gb_auc,
        "Train_Time_s": gb_time
    })

    feature_importances["Gradient Boosting"] = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": best_grad_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    model_preds["Gradient Boosting"] = test_preds
    model_probs["Gradient Boosting"] = test_probs
    model_params["Gradient Boosting"] = gd_study.best_params


    # ------------------------------------------
    # MODEL 4: Random Forest (Adaboost)
    # ------------------------------------------
    # Execute Search
    ada_study = optuna.create_study(direction="maximize")
    ada_study.optimize(lambda trial: adaboost_objective(trial, X_train_large, y_train_large), n_trials=num_trials)

    # Visualize tradeoffs in AdaBoost validation
    vis.plot_parallel_coordinate(ada_study).write_image("phases/results/RF/rf_ada_parallel.png")
    vis.plot_optimization_history(ada_study).write_image("phases/results/RF/rf_ada_optimization_history.png")
    vis.plot_contour(ada_study, params=["max_depth", "n_estimators"]).write_image("phases/results/RF/rf_ada_contour.png")

    print(f"Best AdaBoost CV ROC-AUC: {ada_study.best_value:.4f}")
    print(f"Best Params: {ada_study.best_params}")
    
    # Reconstruct best estimator params
    best_max_depth = ada_study.best_params["max_depth"]
    best_n_estimators = ada_study.best_params["n_estimators"]
    best_lr = ada_study.best_params["learning_rate"]
    best_weight = ada_study.best_params["class_weight"]

    best_ada_model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=best_max_depth, 
                                         random_state=RANDOM_STATE, 
                                         class_weight=best_weight),
        learning_rate=best_lr,
        n_estimators=best_n_estimators,
        random_state=RANDOM_STATE,
    )
    start = time.perf_counter()
    best_ada_model.fit(X_train_large, y_train_large)
    ada_time = time.perf_counter() - start

    test_preds = best_ada_model.predict(X_test)
    test_probs = best_ada_model.predict_proba(X_test)[:, 1]

    ada_acc = accuracy_score(y_test, test_preds)
    ada_f1 = f1_score(y_test, test_preds)
    ada_auc = roc_auc_score(y_test, test_probs)
    ada_precision = precision_score(y_test, test_preds)
    ada_recall = recall_score(y_test, test_preds)

    print(f"Final AdaBoost Test Accuracy: {ada_acc:.4f}")
    print(f"Final AdaBoost Test ROC-AUC: {ada_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, test_preds))

    results.append({
        "Model": "AdaBoost",
        "Accuracy": ada_acc,
        "F1": ada_f1,
        "Precision": ada_precision,
        "Recall": ada_recall,
        "ROC_AUC": ada_auc,
        "Train_Time_s": ada_time
    })

    feature_importances["AdaBoost"] = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": best_ada_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    model_preds["AdaBoost"] = test_preds
    model_probs["AdaBoost"] = test_probs
    model_params["AdaBoost"] = ada_study.best_params


    # Show all results in dataframe
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values(
        by="ROC_AUC",
        ascending=False
    )

    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON")
    print("=" * 80)

    print(comparison_df.round(4).to_string(index=False))

    # Save results of the winning model, with all variants (like in svm.py)
    best_model_name = comparison_df.iloc[0]["Model"]

    # One variant entry per model, each carrying its own best params
    variants = []
    for row in results:
        variant = dict(row)
        variant["best_params"] = model_params[row["Model"]]
        variants.append(variant)

    save_results(
        "random_forest",
        y_test,
        model_preds[best_model_name],
        model_probs[best_model_name],
        best_params=model_params[best_model_name],
        variants=variants,
    )

    #Feature importance results
    print("\n" + "=" * 80)
    print("TOP 15 FEATURE IMPORTANCES")
    print("=" * 80)

    for model_name, importance_df in feature_importances.items():

        print(f"\n{model_name}")
        print("-" * 50)

        print(
            importance_df.head(15)
            .round(4)
            .to_string(index=False)
        )




if __name__ == "__main__":
    run()