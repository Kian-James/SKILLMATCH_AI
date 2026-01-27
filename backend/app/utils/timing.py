# backend/app/utils/timing.py
import time
from functools import wraps
from app.core.logging import log_info

def measure_time(func):
    """
    Decorator to measure function execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        log_info(f"{func.__name__} executed in {end - start:.4f} seconds")
        return result
    return wrapper
