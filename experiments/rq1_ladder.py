"""RQ1: run the model ladder in every setting (S->S, R->R, S->R, R->S) and save to the store.

Full-size runs hold the cross-process heavy-job lock (airsat.locking) from data loading to the
last save, so they never overlap another memory-heavy experiment. Dev runs (--sample) skip it.

Examples:
    python experiments/rq1_ladder.py --models logreg lgbm --sample 30000   # quick dev run
    python experiments/rq1_ladder.py --models xgb                          # full run, locked
    python experiments/rq1_ladder.py --models all --no-lock                # full run, no lock
"""
import argparse
import gc
import logging
from contextlib import nullcontext

from airsat.locking import heavy_job
from airsat.models import REGISTRY, get_spec
from airsat.runner import DIRECTIONS, load_datasets, run_and_save

log = logging.getLogger("rq1_ladder")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["all"], help=f"any of {list(REGISTRY)} or 'all'")
    parser.add_argument("--directions", nargs="+", default=list(DIRECTIONS), choices=DIRECTIONS)
    parser.add_argument("--sample", type=int, default=None, help="rows per dataset for a dev run")
    parser.add_argument("--lock", action=argparse.BooleanOptionalAction, default=None,
                        help="hold the heavy-job lock (default: on for full runs, off for --sample)")
    return parser.parse_args(argv)


def model_names(requested: list[str]) -> list[str]:
    names = list(REGISTRY) if requested == ["all"] else requested
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown}; available: {list(REGISTRY)}")
    return names


def use_lock(args: argparse.Namespace) -> bool:
    return args.lock if args.lock is not None else args.sample is None


def run_models(names: list[str], directions: list[str], sample: int | None) -> None:
    data = load_datasets(sample)
    for name in names:
        log.info("running %s (sample=%s, directions=%s)", name, sample, directions)
        run_and_save(get_spec(name), data, directions, sample)
        gc.collect()


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)
    names = model_names(args.models)
    guard = heavy_job(f"rq1_ladder:{'+'.join(names)}") if use_lock(args) else nullcontext()
    with guard:
        run_models(names, args.directions, args.sample)


if __name__ == "__main__":
    main()
