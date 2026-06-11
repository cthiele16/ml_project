import json
import importlib
from pathlib import Path
import pandas as pd

# folder where the models save their json results
RESULTS_DIR = Path(__file__).parent / "results"

# for each model: name of the json file -> the module we need to run to create it
models = {
    "baseline": "phases.baseline_model",
    "logisticRegression": "phases.logistic_regression_model",
    "svm": "phases.svm",
    "random_forest": "phases.random_forest_experiments",
}


def run():
    rows = []

    # go through all models
    for name, module_path in models.items():
        json_path = RESULTS_DIR / (name + ".json")

        # if the json does not exist yet, run the model first to create it
        if not json_path.exists():
            print(name + ".json not found, running the model first...")
            module = importlib.import_module(module_path)
            module.run()

        # read the results from the json file
        with open(json_path) as f:
            data = json.load(f)

        # only keep the values we want to compare
        rows.append({
            "model": name,
            "accuracy": data["accuracy"],
            "precision": data["precision"],
            "recall": data["recall"],
            "f1": data["f1"],
            "roc_auc": data["roc_auc"],
            "aupr": data["aupr"]
        })

    # put everything in a table and sort by roc_auc (best model on top)
    comparison = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)

    print("\nMODEL COMPARISON")
    print(comparison.round(4).to_string(index=False))

    return comparison


if __name__ == "__main__":
    run()
