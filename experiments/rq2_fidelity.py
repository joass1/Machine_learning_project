"""RQ2: can the synthetic copy of the survey stand in for the real one?

Compares the synthetic training rows (Kaggle S6E10) with the real survey, step by step:
  distortion   per-feature KS (numerics) / TVD (ratings, categoricals), delay summaries and bands,
               Spearman-correlation difference;
  adversarial  LightGBM separating synthetic from real rows (gain importance, one-feature AUCs),
               plus synthetic train vs synthetic test as the "same generator" baseline (~0.50);
  dependence   univariate AUC of each feature for satisfaction, on equal-size samples;
  label_noise  rows where a confident real-trained LightGBM disagrees with the label, synthetic rows
               vs held-out real rows (reuses the stored full lgbm__raw run when present);
  utility      real-data learning curve vs TSTR on real fold 0: how many real rows is the synthetic
               set worth?
  ranking      do the RQ1 model rankings agree across S->S, R->R, S->R, R->S? (>= 3 stored models)
Each step writes its tables and updates rq2_summary.json as soon as it finishes, so a partial run
(--steps / --skip) adds to the results of earlier runs of the same size.

Examples:
    python experiments/rq2_fidelity.py --sample 20000        # dev run -> reports/{results,figures}/dev/
    python experiments/rq2_fidelity.py                       # full run; waits for the heavy-job lock
    python experiments/rq2_fidelity.py --skip utility        # everything except the learning curve/TSTR
"""
import argparse
import gc
import json
import logging
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # figures are only saved; no display in background runs

from airsat import fidelity as fd  # noqa: E402
from airsat import fidelity_plots as fp  # noqa: E402
from airsat import fidelity_utility as fu  # noqa: E402
from airsat import plotstyle as ps  # noqa: E402
from airsat.config import (CATEGORICAL, FEATURES, FIG_DIR, OOF_DIR, RESULTS_DIR, ROOT, SEED,  # noqa: E402
                           SYNTHETIC_DIR, TARGET)
from airsat.data import load_real, to_model_frame  # noqa: E402
from airsat.folds import get_folds, make_folds  # noqa: E402
from airsat.locking import heavy_job  # noqa: E402
from airsat.metrics import auc  # noqa: E402

log = logging.getLogger("rq2")
STEPS = ("distortion", "adversarial", "dependence", "label_noise", "utility", "ranking")
DATA_STEPS = set(STEPS) - {"ranking"}  # ranking only reads the prediction store
UTILITY_TEST_FOLD = 0
STORED_REFERENCE = "lgbm__raw"  # RQ1 real-trained run reused for the label-noise estimate
MIN_RANKED_MODELS = 3
TOP_K = 5
NOISE_COLUMNS = ["syn_confident_share", "syn_flagged_share", "real_confident_share", "real_flagged_share",
                 "ratio"]


@dataclass(frozen=True)
class Dataset:
    X: pd.DataFrame  # model frame: FEATURES with fixed category codes
    y: np.ndarray

    def with_target(self) -> pd.DataFrame:
        return self.X.assign(**{TARGET: self.y})


@dataclass(frozen=True)
class Context:
    syn: Dataset | None
    real: Dataset | None
    real_folds: np.ndarray | None
    sample: int | None
    fig_dir: Path


@dataclass(frozen=True)
class StepOutput:
    tables: dict[str, pd.DataFrame]
    summary: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=None, help="rows per dataset for a dev run")
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS), help="steps to run (default: all)")
    parser.add_argument("--skip", nargs="+", choices=STEPS, default=[], help="steps to leave out")
    return parser.parse_args()


# ---------- data ----------
def _subsample(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
    return df if n is None or n >= len(df) else df.sample(n, random_state=SEED).reset_index(drop=True)


def _read_synthetic(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """One synthetic file (train.csv or test.csv), categoricals read as category to save memory."""
    return pd.read_csv(SYNTHETIC_DIR / name, usecols=columns, dtype={c: "category" for c in CATEGORICAL})


def _read_synthetic_train() -> pd.DataFrame:
    train = _read_synthetic("train.csv", [*FEATURES, TARGET])
    return train.assign(**{TARGET: train[TARGET].astype(int)})


def load(sample: int | None) -> tuple[Dataset, Dataset]:
    """Synthetic training rows and the real survey (test.csv is only read by the adversarial baseline)."""
    out = []
    for name, reader in (("synthetic", _read_synthetic_train), ("real", load_real)):
        df = _subsample(reader(), sample)
        out.append(Dataset(to_model_frame(df), df[TARGET].to_numpy()))
        log.info("%s: %d rows, %.1f%% satisfied", name, len(df), 100 * df[TARGET].mean())
        del df
        gc.collect()
    return out[0], out[1]


def make_context(sample: int | None, fig_dir: Path, needs_data: bool) -> Context:
    if not needs_data:
        return Context(None, None, None, sample, fig_dir)
    syn, real = load(sample)
    real_folds = make_folds(real.y) if sample else get_folds("real", real.y)
    return Context(syn, real, real_folds, sample, fig_dir)


# ---------- steps ----------
def _top(table: pd.DataFrame, key: str, value: str, k: int = TOP_K) -> dict:
    head = table.head(k)
    return dict(zip(head[key].astype(str), head[value].round(4)))


def distortion_step(ctx: Context) -> StepOutput:
    """Per-feature KS/TVD, delay summaries and bands, and the Spearman-correlation difference."""
    syn, real = ctx.syn, ctx.real
    distortion = fd.marginal_distortion(syn.X, real.X)
    delays, bins = fd.delay_summary(syn.X, real.X), fd.delay_bins(syn.X, real.X)
    correlation = fd.correlation_difference(syn.with_target(), real.with_target())
    fp.plot_distortion(distortion, ctx.fig_dir / "rq2_distortion.png")
    fp.plot_delays(bins, ctx.fig_dir / "rq2_delays.png")
    log.info("distortion: most distorted %s", _top(distortion, "feature", "distortion", 3))
    share_zero = delays.pivot(index="feature", columns="dataset", values="share_zero").round(4)
    summary = {"exact_copies": fd.exact_copies(syn.X, real.X),
               "top_distortion": _top(distortion, "feature", "distortion"),
               "mean_distortion_by_kind": distortion.groupby("kind")["distortion"].mean().round(4).to_dict(),
               "delay_share_zero": share_zero.to_dict("index"),
               "largest_correlation_gaps": correlation.head(TOP_K).round(4).to_dict("records"),
               "mean_abs_correlation_gap": round(float(correlation["diff"].abs().mean()), 4)}
    tables = {"distortion": distortion, "delay_summary": delays, "delay_bins": bins,
              "correlation_difference": correlation}
    return StepOutput(tables, summary)


def _train_vs_test(syn: Dataset, sample: int | None) -> fd.AdversarialResult:
    """Synthetic train vs synthetic test rows: two samples of one generator, so AUC should be ~0.50."""
    n = min(fd.ADVERSARIAL_N, sample or fd.ADVERSARIAL_N)
    test = to_model_frame(_subsample(_read_synthetic("test.csv", FEATURES), n))
    gc.collect()
    result = fd.adversarial_auc(syn.X, test, features=FEATURES)
    del test
    gc.collect()
    return result


def _adversarial_summary(result: fd.AdversarialResult) -> dict:
    return {"auc": result.auc, "fold_aucs": result.fold_aucs, "n_per_class": result.n_per_class}


def adversarial_step(ctx: Context) -> StepOutput:
    """Detectability: synthetic vs real (with gain importance and one-feature AUCs) and train vs test."""
    syn_vs_real = fd.adversarial_auc(ctx.syn.X, ctx.real.X, features=FEATURES)
    single = fd.single_feature_adversarial(ctx.syn.X, ctx.real.X)
    baseline = _train_vs_test(ctx.syn, ctx.sample)
    importance = syn_vs_real.importance.rename_axis("feature").reset_index(name="gain_share")
    overall = pd.DataFrame([{"comparison": name, **_adversarial_summary(result)}
                            for name, result in (("syn_vs_real", syn_vs_real), ("train_vs_test", baseline))])
    log.info("adversarial: synthetic vs real %.4f, synthetic train vs test %.4f", syn_vs_real.auc, baseline.auc)
    summary = {"syn_vs_real": {**_adversarial_summary(syn_vs_real),
                               "top_gain_share": _top(importance, "feature", "gain_share")},
               "train_vs_test": _adversarial_summary(baseline),
               "top_single_feature_auc": _top(single, "feature", "auc")}
    tables = {"adversarial": overall.assign(fold_aucs=overall["fold_aucs"].map(lambda f: ";".join(map(str, f)))),
              "adversarial_importance": importance, "adversarial_single_feature": single}
    return StepOutput(tables, summary)


def dependence_step(ctx: Context) -> StepOutput:
    """Univariate AUC of each feature for satisfaction, synthetic vs real, on equal-size samples."""
    dependence = fd.compare_target_dependence(ctx.syn.with_target(), ctx.real.with_target())
    fp.plot_target_dependence(dependence, ctx.fig_dir / "rq2_target_dependence.png")
    strongest = {name: dependence.loc[dependence[col].idxmax(), "feature"]
                 for name, col in (("synthetic", "syn_auc"), ("real", "real_auc"))}
    log.info("dependence: strongest single feature %s", strongest)
    gaps = dependence.head(TOP_K)[["feature", "syn_auc", "real_auc", "diff"]].round(4)
    summary = {"n_rows": int(dependence["n_rows"].iloc[0]), "strongest_feature": strongest,
               "largest_gaps": gaps.to_dict("records"),
               "mean_abs_gap": round(float(dependence["diff"].abs().mean()), 4)}
    return StepOutput({"target_dependence": dependence}, summary)


def _real_reference(ctx: Context) -> tuple[np.ndarray, np.ndarray, str]:
    """Real OOF predictions + real-model predictions on synthetic rows: stored full run, else a refit."""
    syn, real = ctx.syn, ctx.real
    stored = None if ctx.sample else fu.stored_real_predictions(len(real.y), len(syn.y), key=STORED_REFERENCE)
    if stored is not None:
        oof, p_syn, meta = stored
        log.info("label noise: reusing stored %s (created %s)", STORED_REFERENCE, meta.get("created"))
        return oof, p_syn, f"store:{STORED_REFERENCE}"
    reason = "dev run" if ctx.sample else f"no full {STORED_REFERENCE} run in the store"
    log.info("label noise: %s; fitting the 5-fold real reference model", reason)
    result = fu.real_reference_predictions(real.X, real.y, ctx.real_folds, syn.X, syn.y)
    return result.oof, result.applied["synthetic"], f"refit:{result.key}"


def label_noise_step(ctx: Context) -> StepOutput:
    """Confidently 'wrong' labels by a real-trained model: synthetic rows vs held-out real rows."""
    oof, p_syn, source = _real_reference(ctx)
    noise = fd.compare_label_noise(ctx.syn.y, p_syn, ctx.real.y, oof)
    by_threshold = {f"{t:g}": row.to_dict()
                    for t, row in noise.set_index("threshold")[NOISE_COLUMNS].round(5).iterrows()}
    summary = {"source": source, "real_oof_auc": auc(ctx.real.y, oof),
               "real_model_on_synthetic_auc": auc(ctx.syn.y, p_syn), "by_threshold": by_threshold}
    log.info("label noise: flagged share (threshold: synthetic vs real) %s",
             {t: f"{v['syn_flagged_share']:.4f} vs {v['real_flagged_share']:.4f}" for t, v in by_threshold.items()})
    return StepOutput({"label_noise": noise}, summary)


def utility_step(ctx: Context) -> StepOutput:
    """Real learning curve vs TSTR on one held-out real fold; how many real rows the synthetic set is worth."""
    syn, real = ctx.syn, ctx.real
    res = fu.utility_curve(real.X, real.y, ctx.real_folds, syn.X, syn.y, test_fold=UTILITY_TEST_FOLD)
    tstr = (res.tstr.estimate, res.tstr.low, res.tstr.high)
    fp.plot_utility(res.summary, tstr, res.equivalent_n, res.n_synthetic, ctx.fig_dir / "rq2_utility.png")
    trtr = res.curve.iloc[-1]  # the real-trained fit the paired bootstrap compares TSTR with
    log.info("utility: TSTR %.5f vs real n=%d %.5f; equivalent real rows %s", res.tstr.estimate, trtr["n"],
             trtr["auc"], "beyond every size tried" if res.equivalent_n is None else f"{res.equivalent_n:.0f}")
    summary = {"test_fold": UTILITY_TEST_FOLD, "n_test": res.n_test, "n_pool": res.n_pool,
               "n_synthetic": res.n_synthetic, "tstr": asdict(res.tstr), "tstr_trees": res.tstr_trees,
               "trtr": {"n": int(trtr["n"]), "seed": int(trtr["seed"]), "auc": float(trtr["auc"])},
               "tstr_minus_trtr": asdict(res.tstr_minus_trtr), "equivalent_n": res.equivalent_n,
               "equivalent_n_ci": list(res.equivalent_n_ci), "tstr_beats_every_real_size": res.equivalent_n is None,
               "curve_auc_mean": dict(zip(res.summary["n"].astype(str), res.summary["auc_mean"].round(5)))}
    return StepOutput({"utility_curve": res.curve, "utility_summary": res.summary}, summary)


def ranking_step(ctx: Context) -> StepOutput:
    """Model-selection fidelity: rank agreement of the stored RQ1 model AUCs across settings."""
    root = OOF_DIR / "dev" if ctx.sample else OOF_DIR
    store = root.relative_to(ROOT).as_posix()
    ladder = fd.ladder_from_store(root, include_sampled=bool(ctx.sample))
    if len(ladder) < MIN_RANKED_MODELS:
        reason = f"{len(ladder)} model(s) in {store}, need >= {MIN_RANKED_MODELS}"
        log.info("ranking skipped: %s", reason)
        return StepOutput({}, {"skipped": reason})
    ranking = fd.ranking_agreement(ladder)
    pairs = {f"{r.setting_a} vs {r.setting_b}": {"n_models": r.n_models, "kendall_tau": r.kendall_tau,
                                                 "spearman_rho": r.spearman_rho, "same_top": r.same_top}
             for r in ranking.itertuples()}
    log.info("ranking: Kendall tau %s", {k: v["kendall_tau"] for k, v in pairs.items()})
    return StepOutput({"ladder": ladder, "ranking_agreement": ranking},
                      {"store": store, "n_models": len(ladder), "pairs": pairs})


STEP_FUNCTIONS = {"distortion": distortion_step, "adversarial": adversarial_step, "dependence": dependence_step,
                  "label_noise": label_noise_step, "utility": utility_step, "ranking": ranking_step}


# ---------- outputs ----------
def _json_default(value):
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.floating):
        return round(float(value), 6)
    raise TypeError(f"not JSON serialisable: {type(value)}")


def _previous_summary(path: Path, sample: int | None) -> dict:
    """Earlier results of the same run size, so partial runs (--steps / --skip) accumulate."""
    if not path.exists():
        return {}
    try:
        previous = json.loads(path.read_text())
    except json.JSONDecodeError:
        log.warning("ignoring unreadable %s", path)
        return {}
    return previous if previous.get("sample") == sample else {}


def write_step(step: str, out: StepOutput, meta: dict, results_dir: Path) -> None:
    for name, table in out.tables.items():
        table.to_csv(results_dir / f"rq2_{name}.csv", index=False)
    path = results_dir / "rq2_summary.json"
    summary = {**_previous_summary(path, meta["sample"]), **meta, step: out.summary}
    path.write_text(json.dumps(summary, indent=2, default=_json_default))
    log.info("%s: wrote %d table(s) + summary to %s", step, len(out.tables), results_dir)


def run_meta(ctx: Context) -> dict:
    meta = {"sample": ctx.sample, "params": {"adversarial": fd.ADVERSARIAL_PARAMS, "adversarial_n": fd.ADVERSARIAL_N,
                                             "lgbm": fu.FIDELITY_LGBM, "utility_sizes": fu.UTILITY_SIZES,
                                             "utility_seeds": fu.UTILITY_SEEDS}}
    if ctx.syn is not None:
        meta["n_rows"] = {"synthetic": len(ctx.syn.y), "real": len(ctx.real.y)}
    return meta


def run(args: argparse.Namespace) -> None:
    start = time.time()
    results_dir = RESULTS_DIR / "dev" if args.sample else RESULTS_DIR
    fig_dir = FIG_DIR / "dev" if args.sample else FIG_DIR
    for folder in (results_dir, fig_dir):
        folder.mkdir(parents=True, exist_ok=True)
    steps = [s for s in STEPS if s in args.steps and s not in args.skip]
    ctx = make_context(args.sample, fig_dir, needs_data=bool(DATA_STEPS.intersection(steps)))
    meta = run_meta(ctx)
    ps.apply()  # fidelity_plots draws in the house style but does not switch it on itself
    for step in steps:
        step_start = time.time()
        log.info("step %s", step)
        out = STEP_FUNCTIONS[step](ctx)
        timed = StepOutput(out.tables, {**out.summary, "seconds": round(time.time() - step_start, 1)})
        write_step(step, timed, meta, results_dir)
        del out, timed
        gc.collect()
    log.info("done (%s) in %.0fs", ", ".join(steps), time.time() - start)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    with nullcontext() if args.sample else heavy_job("rq2_fidelity"):
        run(args)


if __name__ == "__main__":
    main()
