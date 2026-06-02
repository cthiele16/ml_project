from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, roc_auc_score
from preprocessing import RANDOM_STATE, run as preprocess

def preprocess_special():
    X_train, y_train, X_val, X_test, y_val, y_test, preprocessor = preprocess()

    return X_train, X_test, y_train, y_test

def run():
    """
    Random baseline model
    """
    X_train, X_test, y_train, y_test = preprocess_special()

    # Setup the Random Baseline (Uniform strategy)
    random_model = DummyClassifier(strategy='uniform', random_state=RANDOM_STATE)
    random_model.fit(X_train, y_train)

    # Make predictions
    y_pred_random = random_model.predict(X_test)
    y_prob_random = random_model.predict_proba(X_test)[:, 1]

    # Evaluate
    print("--- Random Baseline Results ---")
    print(f"AUC Score: {roc_auc_score(y_test, y_prob_random):.3f}")
    print(classification_report(y_test, y_pred_random))

if __name__ == "__main__":
    run()