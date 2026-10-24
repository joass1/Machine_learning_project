"""RQ1 report: ladder table in four settings, paired deltas of each technique, and the figures.

Reads only the prediction store, so run it after rq1_ladder.py (and rq1_stacking.py for ensembles).

    python experiments/rq1_report.py            # full runs  -> reports/{results,figures}/
    python experiments/rq1_report.py --dev      # dev store  -> reports/{results,figures}/dev/
"""
import argparse
import json
import logging

import matplotlib

matplotlib.use("Agg")

from airsat import plotstyle  # noqa: E402
from airsat.config import FIG_DIR, OOF_DIR, RESULTS_DIR, TARGET  # noqa: E402
from airsat.data import load_real, load_synthetic  # noqa: E402
from airsat.ladder import ladder_table, trick_deltas  # noqa: E402
from airsat.ladder_plots import plot_ladder, plot_tricks  # noqa: E402
from airsat.runner import load_datasets  # noqa: E402

log = logging.getLogger("rq1_report")


def targets(dev_sample: int | None) -> dict:
    """Labels in store order; dev runs sampled rows with the runner's seed, so resample the same way."""
    if dev_sample:
        data = load_datasets(dev_sample)
        return {"synthetic": data.y_syn, "real": data.y_real}
    train, _ = load_synthetic()
    return {"synthetic": train[TARGET].to_numpy(), "real": load_real()[TARGET].to_numpy()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev", type=int, nargs="?", const=20000, default=None,
                        help="read artifacts/oof/dev (runs made with --sample N; default N = 20000)")
    parser.add_argument("--n-boot", type=int, default=200)
    args = parser.parse_args()
    root = OOF_DIR / "dev" if args.dev else OOF_DIR
    results_dir, fig_dir = (RESULTS_DIR / "dev", FIG_DIR / "dev") if args.dev else (RESULTS_DIR, FIG_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    table = ladder_table(root, include_sampled=bool(args.dev))
    table.to_csv(results_dir / "rq1_ladder.csv")
    log.info("ladder\n%s", table.round(5).to_string())
    deltas = trick_deltas(targets(args.dev), root=root, n_boot=args.n_boot)
    deltas.to_csv(results_dir / "rq1_tricks.csv", index=False)
    log.info("tricks\n%s", deltas.round(5).to_string(index=False))
    (results_dir / "rq1_summary.json").write_text(json.dumps({
        "n_models": len(table), "best": {s: table[s].idxmax() for s in ("S->S", "R->R", "S->R", "R->S")},
        "n_boot": args.n_boot, "dev_sample": args.dev}, indent=2))

    plotstyle.apply()
    plot_ladder(table, fig_dir / "rq1_ladder.png")
    if not deltas.empty:
        plot_tricks(deltas, fig_dir / "rq1_tricks.png")
    log.info("wrote %s and %s", results_dir, fig_dir)


if __name__ == "__main__":
    main()
