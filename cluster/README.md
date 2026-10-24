# Running on the SCIS Origami GPU cluster

Full-size experiments are too large for a laptop with 17 GB RAM, so they run as SLURM jobs on the
SCIS GPU cluster. `cluster/remote.py` wraps the steps; it only needs `ssh`/`scp` on the PATH.

## One-time setup

1. Connect to the SMU GlobalProtect VPN.
2. Point the helper at the group account (account name and cluster address are in the SCIS
   cluster email; they are deliberately not stored in this repo):
   ```powershell
   $env:AIRSAT_CLUSTER = "<group account>@<cluster address>"     # bash: export AIRSAT_CLUSTER=...
   ```
3. Key-based SSH (password login works too, but the helper uses keys):
   ```powershell
   ssh-keygen -t ed25519            # if you have no key yet
   type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh -o PubkeyAuthentication=no $env:AIRSAT_CLUSTER "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys && chcon -R -t ssh_home_t ~/.ssh"
   ```
   The `chcon` matters: without the SELinux label `ssh_home_t` the server ignores `authorized_keys`.
4. Push code and data, then build the virtual environment (Python 3.11 module + torch 2.6 CUDA 12.4):
   ```
   python cluster/remote.py push --data
   python cluster/remote.py setup
   ```

Use a different key with `AIRSAT_SSH_KEY=path`.

## Running a job

```
python cluster/remote.py push                      # after every code change
python cluster/remote.py submit --name ladder --wait -- python experiments/rq1_ladder.py --models xgb
python cluster/remote.py status
python cluster/remote.py fetch                     # artifacts/oof, artifacts/folds, reports/{results,figures}
```

`--gpus` picks the card: `1` (any free GPU, default), `3090:1`, `a40:1` or `l40s:1` (48 GB, only
needed for TabPFN with a large context). `--gpus 0` asks for a CPU-only job.

## Cluster limits (account `is460`, QoS `is460qos`)

- One running and two submitted jobs per user, 1 day wall time each. `submit` retries every minute
  while the submit limit is reached; a second job waits in the queue until the first finishes.
- Home: 180 GB. `~/scratchDirectory` is purged after 30 days, so results live under `~/airsat/repo`.
- Remote layout: `~/airsat/repo` (code + data + outputs), `~/airsat/venv`, `~/airsat/jobs`
  (sbatch scripts and `airsat-<name>-<jobid>.out` logs).
- Jobs clear `artifacts/locks/heavy.lock` at start: the one-job limit already prevents overlap,
  and a job killed at its time limit would otherwise leave the lock behind.
