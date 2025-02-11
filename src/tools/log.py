import logging
import time
from functools import wraps

# Setting up logging for detailed output
logging.basicConfig(level=logging.INFO, format="d-%(levelname)s-%(message)s")


def timing(msg):
    def decorator(f):
        @wraps(f)
        def wrap(*args, **kw):
            ts = time.time()
            result = f(*args, **kw)
            te = time.time()
            print(f"{msg} executed in: {te-ts:.4f} sec")
            return result

        return wrap

    return decorator


def print_estimated_time(
    index: int, total: int, start_time: float, intervals_logged: int
) -> int:
    """
    Prints progress and estimated time remaining at intervals of 1/10 of the total.
    """
    elapsed_time = time.time() - start_time
    time_per_iteration = elapsed_time / index
    estimated_time_remaining = (total - index) * time_per_iteration

    if index >= (intervals_logged + 1) * (total // 10) or intervals_logged == 0:
        progress = f"{index}/{total}"
        logging.info(
            f"Progress: {progress} - Elapsed time: {elapsed_time:.2f}s - "
            f"Estimated time remaining: {estimated_time_remaining:.2f}s"
        )
        intervals_logged += 1

    return intervals_logged
