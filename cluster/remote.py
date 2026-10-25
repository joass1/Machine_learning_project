"""Run airsat experiments on the SCIS Origami GPU cluster (SLURM) from a laptop.

Requires the GlobalProtect VPN and key-based SSH (see cluster/README.md). No credentials live here.

    python cluster/remote.py push [--data]          # copy code (and data) to ~/airsat/repo
    python cluster/remote.py setup                  # build ~/airsat/venv (CPU job)
    python cluster/remote.py submit --name ladder -- python experiments/rq1_ladder.py --models xgb
    python cluster/remote.py wait JOBID             # block until the job finishes, print its log
    python cluster/remote.py fetch                  # copy artifacts/ and reports/ back
    python cluster/remote.py status
"""
import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = os.environ.get("AIRSAT_CLUSTER", "")  # "<group account>@<cluster address>", from the SCIS email
KEY = Path(os.environ.get("AIRSAT_SSH_KEY", Path.home() / ".ssh" / "id_ed25519"))
REMOTE = "airsat"                      # relative to the cluster home directory
CODE_PATHS = ["src", "experiments", "tests", "cluster", "pyproject.toml", "README.md",
              "assets/fonts"]  # figure font (local-only, skipped when absent)
DATA_PATHS = ["data/raw/train.csv", "data/raw/test.csv", "data/original_official/data.csv"]
RESULT_PATHS = ["artifacts/oof", "artifacts/folds", "reports/results", "reports/figures"]
SLURM_DEFAULTS = {"partition": "project", "account": "is460", "qos": "is460qos"}
PYTHON_MODULE = "Python/3.11.11-GCCcore-13.3.0"
POLL_SECONDS = 30
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-i", str(KEY)]


def ssh(command: str, check: bool = True) -> subprocess.CompletedProcess:
    if not TARGET:
        raise SystemExit("set AIRSAT_CLUSTER=<group account>@<cluster address> (see cluster/README.md)")
    return subprocess.run(["ssh", *SSH_OPTS, TARGET, command], capture_output=True, text=True, check=check)


def scp(src: str, dst: str) -> None:
    subprocess.run(["scp", "-q", *SSH_OPTS, src, dst], check=True)


def _upload_tar(paths: list[str], label: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / f"{label}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for rel in [p for p in paths if (ROOT / p).exists()]:
                tar.add(ROOT / rel, arcname=rel,
                        filter=lambda info: None if "__pycache__" in info.name else info)
        scp(str(archive), f"{TARGET}:{REMOTE}/{label}.tar.gz")
    # Windows tar entries mark directories read-only; restore write access for pip and outputs.
    ssh(f"cd {REMOTE}/repo && tar -xzf ../{label}.tar.gz && rm ../{label}.tar.gz && chmod -R u+w .")
    print(f"pushed {label}: {', '.join(paths)}")


def push(include_data: bool) -> None:
    ssh(f"mkdir -p {REMOTE}/repo {REMOTE}/jobs")
    _upload_tar(CODE_PATHS, "code")
    if include_data:
        _upload_tar(DATA_PATHS, "data")


def sbatch_script(name: str, command: str, gpus: str, cpus: int, mem: str, time_limit: str,
                  activate: bool = True) -> str:
    gres = f"#SBATCH --gres=gpu:{gpus}\n" if gpus != "0" else ""
    venv = f"source ~/{REMOTE}/venv/bin/activate\n" if activate else ""
    return (
        "#!/bin/bash\n"
        f"#SBATCH --job-name=airsat-{name}\n"
        + "".join(f"#SBATCH --{k}={v}\n" for k, v in SLURM_DEFAULTS.items())
        + gres
        + f"#SBATCH --cpus-per-task={cpus}\n#SBATCH --mem={mem}\n#SBATCH --time={time_limit}\n"
        "#SBATCH --output=%x-%j.out\n"
        "set -euo pipefail\n"
        f"module purge\nmodule load {PYTHON_MODULE}\n"
        f"{venv}cd ~/{REMOTE}/repo\n"
        # The QoS runs one job per user at a time, so a lock left here was orphaned by a killed job.
        "rm -f artifacts/locks/heavy.lock\n"
        'echo "node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} start=$(date)"\n'
        f"{command}\n"
        'echo "end=$(date)"\n'
    )


def submit(name: str, command: str, gpus: str = "1", cpus: int = 8, mem: str = "48G",
           time_limit: str = "06:00:00", activate: bool = True) -> str:
    """Write and submit a job; retries while the QoS submit limit (2 jobs) is reached."""
    script = sbatch_script(name, command, gpus, cpus, mem, time_limit, activate)
    remote_file = f"{REMOTE}/jobs/{name}.sbatch"
    ssh(f"cat > {remote_file} <<'EOF'\n{script}EOF")
    while True:
        result = ssh(f"cd {REMOTE}/jobs && sbatch --parsable {name}.sbatch", check=False)
        if result.returncode == 0:
            job_id = result.stdout.strip().split(";")[0]
            print(f"submitted {name} as job {job_id}")
            return job_id
        if "QOSMaxSubmitJobPerUserLimit" not in result.stderr:
            raise SystemExit(f"sbatch failed: {result.stderr.strip()}")
        print("submit limit reached; retrying in 60 s")
        time.sleep(60)


def wait(job_id: str) -> str:
    """Poll until the job leaves the queue; print and return its final state and log."""
    while ssh(f"squeue -h -j {job_id}", check=False).stdout.strip():
        time.sleep(POLL_SECONDS)
    state = ssh(f"sacct -n -X -j {job_id} -o State%20", check=False).stdout.strip()
    log = ssh(f"cat {REMOTE}/jobs/*-{job_id}.out", check=False).stdout
    print(f"job {job_id}: {state}\n{log[-4000:]}")
    return state


def fetch() -> None:
    existing = [p for p in RESULT_PATHS if ssh(f"test -e {REMOTE}/repo/{p}", check=False).returncode == 0]
    if not existing:
        print("nothing to fetch")
        return
    ssh(f"cd {REMOTE}/repo && tar -czf ../results.tar.gz {' '.join(existing)}")
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "results.tar.gz"
        scp(f"{TARGET}:{REMOTE}/results.tar.gz", str(local))
        with tarfile.open(local) as tar:
            tar.extractall(ROOT, filter="data")
    print(f"fetched: {', '.join(existing)}")


# torch is pinned to a CUDA 12.4 wheel (driver 590 on the nodes); the constraint stops pytabkit or
# tabpfn from pulling a different torch build.
SETUP = (
    f"python3 -m venv ~/{REMOTE}/venv\nsource ~/{REMOTE}/venv/bin/activate\n"
    "pip install -q --upgrade pip\n"
    "pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124\n"
    "echo torch==2.6.0 > /tmp/airsat-constraints-$$.txt\n"
    "pip install -q -e '.[dev,deep]' --constraint /tmp/airsat-constraints-$$.txt\n"
    "python -c 'import torch, airsat; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())'\n"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("push").add_argument("--data", action="store_true")
    sub.add_parser("setup").add_argument("--gpus", default="0", help="'0' for a CPU-only job")
    sub.add_parser("status")
    sub.add_parser("fetch")
    sub.add_parser("wait").add_argument("job_id")
    p_submit = sub.add_parser("submit")
    p_submit.add_argument("--name", required=True)
    p_submit.add_argument("--gpus", default="1", help="'1' (any), 'l40s:1', 'a40:1', '3090:1' or '0'")
    p_submit.add_argument("--cpus", type=int, default=8)
    p_submit.add_argument("--mem", default="48G")
    p_submit.add_argument("--time", default="06:00:00")
    p_submit.add_argument("--wait", action="store_true", help="wait for the job, then fetch results")
    p_submit.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.cmd == "push":
        push(args.data)
    elif args.cmd == "setup":
        job = submit("setup", SETUP, gpus=args.gpus, cpus=4, mem="16G", time_limit="01:00:00",
                     activate=False)
        wait(job)
    elif args.cmd == "status":
        print(ssh("squeue -u $USER -o '%.10i %.24j %.9T %.10M %.20R'", check=False).stdout)
    elif args.cmd == "fetch":
        fetch()
    elif args.cmd == "wait":
        wait(args.job_id)
    elif args.cmd == "submit":
        command = " ".join(a for a in args.command if a != "--")
        job = submit(args.name, command, args.gpus, args.cpus, args.mem, args.time)
        if args.wait:
            state = wait(job)
            fetch()
            sys.exit(0 if "COMPLETED" in state else 1)


if __name__ == "__main__":
    main()
