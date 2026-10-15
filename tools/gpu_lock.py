from contextlib import contextmanager
from experiment_common import ROOT

@contextmanager
def gpu_lock():
    """Serialize this project's inference/teacher/training processes on one GPU."""
    import fcntl
    with (ROOT/'.gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN)
