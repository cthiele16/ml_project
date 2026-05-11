# ml_project
ML Final Project UPC 

# All projects must include:
● Exploratory Data Analysis (EDA)
● Data preprocessing (missing values, encoding, scaling)
● Feature selection or extraction
● Model comparison
● Interpretation of results
● Discussion of dataset challenges

# Required Project Phases
Your project must include the following phases:

# Phase I — Unsupervised Structural Analysis
● Apply PCA and analyze explained variance
● Perform clustering (K-Means and GMM)
● Justify the number of clusters (Elbow Method / Silhouette Score)
● Discuss whether clusters align with the target variable

# Phase II — Linear and Regularized Models
● Train:
○ Logistic Regression (classification)
○ Ridge / Lasso (regression) -> not needed for our binary problem 
● Study the effect of regularization (λ)
● Analyze how model coefficients change

# Phase III — Kernel Methods (SVM)
● Train SVM models using:
○ Linear kernel
○ Polynomial kernel
○ RBF kernel
● Perform hyperparameter tuning (Grid Search)

# Phase IV — Ensemble Methods
● Train:
○ Random Forest (Bagging)
○ Boosting models (AdaBoost or Gradient Boosting)
● Compare:
○ Performance
○ Training time
○ Feature importance

# Phase V — Model Selection and Validation
● Use cross-validation (preferably nested CV)
● Evaluate using appropriate metrics:

For classification:
● Accuracy
● F1-score
● ROC-AUC
● AUPR

For regression: -> not needed
● R²
● MAE
● RMSE
● Compare models and select the best one
● (Optional) Perform statistical tests (e.g., paired t-test)

# Baseline Models (Mandatory)
You must include at least one simple baseline model to contextualize performance. e.g.
Random Guessing: predicts classes randomly according to their training distribution.
One-Class-prediction: always predict only one class for all samples


1. download dataset in terminal: 
pip install ucimlrepo
python -m pip install matplotlib


How to run: # python main.py phase1
