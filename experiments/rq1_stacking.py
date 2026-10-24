"""RQ1 (ensembles): combine the saved ladder predictions in each training direction.

For each direction (synthetic, real) and method (rank_mean, logistic, hill_climb):
  nested in-dataset AUC (S->S or R->R) and AUC on the other dataset (S->R or R->S).
Also records each combiner's model weights, so we can see whether the synthetic data
would pick the same ensemble as the real data.

    python experiments/rq1_stacking.py                      # uses every full-size run in the store
    python experiments/rq1_stacking.py --exclude tabpfn__raw
"""
import argparse
import json
import logging

import numpy as np
import pandas as pd

from airsat.config import RESULTS_DIR, TARGET
from airsat.data import load_real, load_synthetic
from airsat.folds import get_folds
from airsat.oof import list_runs, load_run
from airsat.stacking import FITTERS, evaluate_stack

log = logging.getLogger("rq1_stacking")
OTHER = {"synthetic": ("real", "apply__real"), "real": ("synthetic", "apply__synthetic")}


def load_targets() -> dict[str, np.ndarray]:
    train, _ = load_synthetic()
    return {"synthetic": train[TARGET].to_numpy(), "real": load_real()[TARGET].to_numpy()}


def common_keys(exclude: list[str]) -> list[str]:
    """Models with full-size runs in both directions."""
    keys = sorted(set(list_runs("synthetic")) & set(list_runs("real")) - set(exclude))
    if len(keys) < 2:
        raise SystemExit(f"need >= 2 models run in both directions, found {keys}")
    return keys


def stack_direction(direction: str, keys: list[str], targets: dict[str, np.ndarray]) -> list[dict]:
    other, apply_name = OTHER[direction]
    runs = [load_run(direction, k)[0] for k in keys]
    Z = np.column_stack([r["oof"] for r in runs])
    Z_apply = np.column_stack([r[apply_name] for r in runs])
    y, y_apply = targets[direction], targets[other]
    folds = get_folds(direction, y)
    rows = []
    for method in FITTERS:
        res = evaluate_stack(Z, y, folds, method, Z_apply, y_apply)
        rows.append({"train": direction, "method": method, "in_dataset_auc": res.nested_auc,
                     "other_dataset_auc": res.apply_auc,
                     "weights": dict(zip(keys, np.round(res.weights, 4).tolist()))})
        log.info("%s %s nested=%.5f other=%.5f", direction, method, res.nested_auc, res.apply_auc)
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exclude", nargs="*", default=[])
    args = parser.parse_args()
    keys, targets = common_keys(args.exclude), load_targets()
    rows = [row for d in ("synthetic", "real") for row in stack_direction(d, keys, targets)]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "rq1_stacking.json").write_text(json.dumps({"models": keys, "rows": rows}, indent=2))
    table = pd.DataFrame(rows).drop(columns="weights")
    table.to_csv(RESULTS_DIR / "rq1_stacking.csv", index=False)
    log.info("\n%s", table.to_string(index=False))


if __name__ == "__main__":
    main()
