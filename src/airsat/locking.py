"""Cross-process lock so only one full-size (memory-heavy) experiment runs at a time.

Usage:
    with heavy_job("rq2_fidelity"):
        ...full-size work...
"""
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

from airsat.config import ARTIFACTS_DIR

log = logging.getLogger(__name__)
LOCK_PATH = ARTIFACTS_DIR / "locks" / "heavy.lock"
POLL_SECONDS = 30


def try_acquire(owner: str, path: Path = LOCK_PATH) -> bool:
    """Atomically create the lock file; False if someone else holds it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as handle:
        handle.write(f"{owner} pid={os.getpid()} since={time.strftime('%H:%M:%S')}\n")
    return True


@contextmanager
def heavy_job(owner: str, path: Path = LOCK_PATH, poll_seconds: float = POLL_SECONDS,
              timeout_seconds: float | None = None):
    """Block until the heavy-job lock is free, hold it for the `with` body, then release it."""
    start = time.time()
    while not try_acquire(owner, path):
        if timeout_seconds is not None and time.time() - start > timeout_seconds:
            raise TimeoutError(f"{owner}: heavy-job lock still held: {path.read_text().strip()}")
        log.info("%s waiting for heavy-job lock held by: %s", owner, path.read_text().strip())
        time.sleep(poll_seconds)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
