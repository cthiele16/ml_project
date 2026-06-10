import argparse
import eda
from phases import logistic_regression_model, baseline_model, random_forest_experiments, svm, modelcomparison
import preprocessing
import unsupervised_structural_analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["eda", "preprocessing", "unsupervised", "baseline", "logistic_regression", "svm", "random_forest", "modelcomparison"])
    args = parser.parse_args()

    if args.phase == "eda":
        eda.run()
    elif args.phase == "preprocessing":
        preprocessing.run()
    elif args.phase == "unsupervised":
        unsupervised_structural_analysis.run()
    elif args.phase == "baseline":
        baseline_model.run()
    elif args.phase == "logistic_regression":
        logistic_regression_model.run()
    elif args.phase == "svm":
        svm.run()
    elif args.phase == "random_forest":
        random_forest_experiments.run()
    elif args.phase == "modelcomparison":
        modelcomparison.run()
