import argparse
import eda
from phases import logistic_regression_model, baseline_model, phase4, svm
import preprocessing
import unsupervised_structural_analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["eda", "preprocessing", "unsupervised", "baseline", "logistic_regression", "svm", "phase4", "phase5"])
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
    elif args.phase == "phase4":
        phase4.run()
    #elif args.phase == "phase5":
        #phase5.run()
