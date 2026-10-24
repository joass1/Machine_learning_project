"""RQ3: which services drive satisfaction per segment, and do synthetic and real data agree?

One LightGBM per dataset is fitted on folds 1-4; every analysis runs on the untouched fold 0:
  (i)   grouped vs single permutation importance (AUC drop) for each model on its own holdout;
  (ii)  TreeSHAP per segment (Class x Type of Travel): synthetic model on synthetic passengers,
        real model on real passengers, and the synthetic model on real passengers;
  (iii) agreement of the service rankings (Spearman rho, top-3 overlap, same #1 "fix-first");
  (iv)  what-if: +1 rating for passengers who rated a service 1-3, both models on real passengers
        and on synthetic passengers. Model-based and associational, not causal.

Examples:
    python experiments/rq3_drivers.py --sample 30000   # dev run -> reports/{results,figures}/dev/
    python experiments/rq3_drivers.py                  # full run; waits for the heavy-job lock
"""
import argparse
import gc
import json
import logging
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # figures are only saved; no display in background runs

from airsat import drivers  # noqa: E402
from airsat.config import (CATEGORICAL, FIG_DIR, RESULTS_DIR, SEED, SERVICE_GROUPS,  # noqa: E402
                           SYNTHETIC_DIR, TARGET)
from airsat.data import load_real, to_model_frame  # noqa: E402
from airsat.drivers_plots import fig_agreement, fig_permutation, fig_shap_heatmap, fig_whatif  # noqa: E402
from airsat.folds import get_folds, make_folds  # noqa: E402
from airsat.locking import heavy_job  # noqa: E402
from airsat.metrics import auc  # noqa: E402

log = logging.getLogger("rq3")
HOLDOUT_FOLD = 0
DATASETS = ("synthetic", "real")
SHAP_VIEWS = {"syn_on_syn": ("synthetic", "synthetic"), "real_on_real": ("real", "real"),
              "syn_on_real": ("synthetic", "real")}  # view -> (model trained on, passengers explained)
COMPARISONS = {"shap_own_data": ("syn_on_syn", "real_on_real"),
               "shap_real_passengers": ("syn_on_real", "real_on_real")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=None, help="rows per dataset for a dev run")
    parser.add_argument("--repeats", type=int, default=3, help="permutation repeats per group/feature")
    parser.add_argument("--shap-rows", type=int, default=drivers.MAX_SHAP_ROWS, help="TreeSHAP rows per segment")
    return parser.parse_args()


def _read_synthetic_train() -> pd.DataFrame:
    """Synthetic training rows only (skips the 470k-row test file to save memory)."""
    train = pd.read_csv(SYNTHETIC_DIR / "train.csv", dtype={c: "category" for c in CATEGORICAL})
    return train.assign(**{TARGET: train[TARGET].astype(int)})


def load(sample: int | None) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    data = {}
    for name, reader in (("synthetic", _read_synthetic_train), ("real", load_real)):
        df = reader()
        if sample and sample < len(df):
            df = df.sample(sample, random_state=SEED).reset_index(drop=True)
        data[name] = (to_model_frame(df), df[TARGET].to_numpy())
        log.info("%s: %d rows, %.1f%% satisfied", name, len(df), 100 * df[TARGET].mean())
        del df
        gc.collect()
    return data


def fit_models(data: dict, sample: int | None) -> tuple[dict, dict]:
    """Driver model per dataset + its holdout (X, y); the full frames are not kept."""
    models, holdouts = {}, {}
    for name in DATASETS:
        X, y = data.pop(name)
        folds = make_folds(y) if sample else get_folds(name, y)
        models[name] = drivers.train_driver_model(X, y, folds, HOLDOUT_FOLD)
        idx = models[name].holdout_idx
        holdouts[name] = (X.iloc[idx].reset_index(drop=True), y[idx])
        del X, y, folds
        gc.collect()
    return models, holdouts


def permutation_tables(models: dict, holdouts: dict, repeats: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    perm, overlap = [], []
    for name in DATASETS:
        X, y = holdouts[name]
        model = models[name].model
        grouped = drivers.permutation_importance_grouped(model, X, y, SERVICE_GROUPS, repeats)
        single = drivers.permutation_importance_single(model, X, y, n_repeats=repeats)
        perm += [grouped.assign(dataset=name, kind="group"), single.assign(dataset=name, kind="single")]
        overlap.append(drivers.single_vs_group(grouped, single, SERVICE_GROUPS).assign(dataset=name))
        log.info("permutation [%s]: groups %s", name, grouped.set_index("name")["auc_drop"].round(4).to_dict())
    return pd.concat(perm, ignore_index=True), pd.concat(overlap, ignore_index=True)


def shap_table(models: dict, holdouts: dict, max_rows: int) -> pd.DataFrame:
    frames = []
    for view, (trained_on, passengers) in SHAP_VIEWS.items():
        start = time.time()
        shap = drivers.segment_shap(models[trained_on].model, holdouts[passengers][0], max_rows=max_rows)
        frames.append(shap.assign(view=view, model=trained_on, passengers=passengers))
        log.info("TreeSHAP [%s]: %d rows in %.0fs", view, shap.groupby("segment")["n_rows"].first().sum(),
                 time.time() - start)
        gc.collect()
    return pd.concat(frames, ignore_index=True)


def whatif_table(models: dict, holdouts: dict) -> pd.DataFrame:
    frames = []
    for passengers in ("real", "synthetic"):
        X = holdouts[passengers][0]
        for trained_on in DATASETS:
            table = drivers.what_if_table(models[trained_on].model, X)
            frames.append(table.assign(model=trained_on, passengers=passengers))
            log.info("what-if [%s model on %s passengers] done", trained_on, passengers)
    return pd.concat(frames, ignore_index=True)


def agreement_table(shap: pd.DataFrame, whatif: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for comparison, (syn_view, real_view) in COMPARISONS.items():
        agree = drivers.ranking_agreement(shap[shap["view"] == syn_view], shap[shap["view"] == real_view])
        frames.append(agree.assign(comparison=comparison, score="mean_abs_shap"))
    ranks = whatif.rename(columns={"service": "feature"})
    for passengers in ("real", "synthetic"):
        on = ranks[ranks["passengers"] == passengers]
        agree = drivers.ranking_agreement(on[on["model"] == "synthetic"], on[on["model"] == "real"],
                                          score="mean_uplift")
        frames.append(agree.assign(comparison=f"whatif_{passengers}_passengers", score="mean_uplift"))
    return pd.concat(frames, ignore_index=True)


def _transfer_auc(models: dict, holdouts: dict, trained_on: str, passengers: str) -> float:
    X, y = holdouts[passengers]
    return auc(y, models[trained_on].model.predict(X))


def summary(models: dict, holdouts: dict, agreement: pd.DataFrame, args, seconds: float) -> dict:
    transfer = {"syn_model_on_real_holdout": _transfer_auc(models, holdouts, "synthetic", "real"),
                "real_model_on_syn_holdout": _transfer_auc(models, holdouts, "real", "synthetic")}
    fix_first = {comp: grp.set_index("segment")[["top1_syn", "top1_real", "top1_match"]].to_dict("index")
                 for comp, grp in agreement.groupby("comparison")}
    return {"sample": args.sample, "holdout_fold": HOLDOUT_FOLD, "permutation_repeats": args.repeats,
            "shap_rows_per_segment_cap": args.shap_rows, "lgbm_params": drivers.driver_params(),
            "models": {name: {"train_rows": int(len(m.train_idx)), "holdout_rows": int(len(m.holdout_idx)),
                              "best_iteration": m.best_iteration, "holdout_auc": m.holdout_auc}
                       for name, m in models.items()},
            "transfer_auc": transfer, "fix_first": fix_first,
            "fix_first_matches": agreement.groupby("comparison")["top1_match"].sum().astype(int).to_dict(),
            "seconds": round(seconds, 1)}


def write_outputs(tables: dict[str, pd.DataFrame], meta: dict, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(results_dir / f"rq3_{name}.csv", index=False)
    (results_dir / "rq3_summary.json").write_text(json.dumps(meta, indent=2, default=_json_default))
    log.info("wrote %d tables + summary to %s", len(tables), results_dir)


def _json_default(value):
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.floating):
        return round(float(value), 6)
    raise TypeError(f"not JSON serialisable: {type(value)}")


def run(args: argparse.Namespace) -> None:
    start = time.time()
    results_dir = RESULTS_DIR / "dev" if args.sample else RESULTS_DIR
    fig_dir = FIG_DIR / "dev" if args.sample else FIG_DIR
    models, holdouts = fit_models(load(args.sample), args.sample)
    perm, overlap = permutation_tables(models, holdouts, args.repeats)
    shap = shap_table(models, holdouts, args.shap_rows)
    whatif = whatif_table(models, holdouts)
    agreement = agreement_table(shap, whatif)
    tables = {"permutation": perm, "single_vs_group": overlap, "segment_shap": shap,
              "whatif": whatif, "agreement": agreement}
    write_outputs(tables, summary(models, holdouts, agreement, args, time.time() - start), results_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_permutation(overlap, fig_dir / "rq3_permutation.png")
    fig_shap_heatmap(shap, fig_dir / "rq3_shap_heatmap.png")
    fig_agreement(agreement, fig_dir / "rq3_agreement.png")
    fig_whatif(whatif, fig_dir / "rq3_whatif.png")
    log.info("done in %.0fs", time.time() - start)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    with nullcontext() if args.sample else heavy_job("rq3_drivers"):
        run(args)


if __name__ == "__main__":
    main()
