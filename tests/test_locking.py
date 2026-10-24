"""Tests for the heavy-job lock."""
import pytest

from airsat.locking import heavy_job, try_acquire


def test_lock_is_exclusive_and_released(tmp_path):
    lock = tmp_path / "heavy.lock"
    with heavy_job("first", path=lock):
        assert lock.exists() and "first" in lock.read_text()
        assert not try_acquire("second", lock)
    assert not lock.exists()
    assert try_acquire("third", lock)


def test_waiting_job_times_out(tmp_path):
    lock = tmp_path / "heavy.lock"
    assert try_acquire("holder", lock)
    with pytest.raises(TimeoutError, match="holder"):
        with heavy_job("waiter", path=lock, poll_seconds=0.01, timeout_seconds=0.05):
            pass


def test_lock_released_on_error(tmp_path):
    lock = tmp_path / "heavy.lock"
    with pytest.raises(RuntimeError):
        with heavy_job("crashy", path=lock):
            raise RuntimeError("boom")
    assert not lock.exists()
