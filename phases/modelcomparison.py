'''
import json
from pathlib import Path
import pandas as pd

rows = []
for path in Path("results").glob("*.json"):
    data = json.load(open(path))
    rows.append({
        "model": path.stem,
        "accuracy": data["accuracy"],
        "f1": data["f1"],
        "roc_auc": data["roc_auc"],
        "aupr": data["aupr"],
    })

comparison = pd.DataFrame(rows).sort_values("aupr", ascending=False)
print(comparison)
'''