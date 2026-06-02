import argparse
import eda
from phases import logistic_regression_model, baseline_model, phase1, phase3, phase4
import preprocessing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["eda", "preprocessing", "phase1", "baseline", "logistic_regression", "phase3", "phase4", "phase5"])
    args = parser.parse_args()

    if args.phase == "eda":
        eda.run()
    elif args.phase == "preprocessing":
        preprocessing.run()
    elif args.phase == "phase1":
        phase1.run()
    elif args.phase == "baseline":
        baseline_model.run()
    elif args.phase == "logistic_regression":
        logistic_regression_model.run()
    elif args.phase == "phase3":
        phase3.run()
    elif args.phase == "phase4":
        phase4.run()
    elif args.phase == "phase5":
        phase5.run()
