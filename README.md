# All projects must include:
* Exploratory Data Analysis (EDA)
* Data preprocessing (missing values, encoding, scaling)
* Feature selection or extraction
* Model comparison
* Interpretation of results
* Discussion of dataset challenges

# Required Project Phases
* Baseline model
* Phase I — Unsupervised Structural Analysis
* Phase II — Linear and Regularized Models
* Phase III — Kernel Methods (SVM)
* Phase IV — Ensemble Methods
* Phase V — Model Selection and Validation

For classification:
* Accuracy
* F1-score
* ROC-AUC
* AUPR

# How to run:
1. download dataset in terminal: 
`pip install ucimlrepo`
`python -m pip install matplotlib`
2. Run the phases (always from the directory `ml_project/`)
`python main.py phasename`
choose between phasenames: "eda", "preprocessing", "unsupervised", "baseline", "logistic_regression", "svm", "random_forest", "modelcomparison"
You can either:
* run all of the phases seperately (1. eda, 2. preprocess, 3. unsupervised, 4. baseline, 5. logistic_regression, 6. svm, 7. random_forest) 
OR 
* you run one time modelcomparison which compares the model results (in the .json files) or, if the model resluts are not calculated yet, fits all the models and gives out the results, compares the models and gives the best model for our dataset depending on the classification metrics Accuracy, F1-score, ROC-AUC and AUPR.
