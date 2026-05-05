from preprocessing_phase2 import X,y
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score

# Phase 2 - Logistic Regression
def preprocess_special():
    print("Start Special Preprocess")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def run():
    # Preprocessing
    X_train, X_test, y_train, y_test = preprocess_special()

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train.values.ravel())

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]


    # Feature importance check 
    importance = pd.DataFrame({'Feature': X.columns, 'Weight': model.coef_[0]})
    print(importance.sort_values(by='Weight', ascending=False))

     # Evaluation metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("AUC Score:", roc_auc_score(y_test, y_prob))

if __name__ == "__main__":
    run()