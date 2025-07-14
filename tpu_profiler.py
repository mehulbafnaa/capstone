# tpu_profiler.py

import contextlib
import jax
import jax.profiler
import time
from pathlib import Path

@contextlib.contextmanager
def profile(log_dir=None):
    """
    A context manager for JAX profiling. Creates a unique log directory
    if one is not provided.

    Usage:
        import tpu_profiler
        with tpu_profiler.profile():
            # Your JAX code to be profiled here
            ...
    """
    # Only have the main process (host 0) control the profiler
    if jax.process_index() == 0:
        if log_dir is None:
            log_dir = f"/tmp/tpu_profiles_{int(time.time())}"
        
        # Ensure the directory exists
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"\n>>> Starting JAX profiler. Traces will be saved to: {log_dir}")
        jax.profiler.start_trace(log_dir)

    try:
        # This 'yield' passes control to the code inside the 'with' block
        yield
    finally:
        # This code runs after the 'with' block is finished, even if there was an error
        if jax.process_index() == 0:
            print(">>> Stopping JAX profiler.")
            jax.profiler.stop_trace()
            print("--- Profiling complete. ---")