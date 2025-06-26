# #!/usr/bin/env python3

# import jax
# import jax.distributed
# import os

# def initialize_tpu():
#     """Initialize TPU v3-16 multi-host setup"""
    
#     # Get TPU environment info
#     host_id = int(os.environ.get('TPU_PROCESS_ID', 0))
#     num_hosts = int(os.environ.get('TPU_PROCESS_COUNT', 2))
#     tpu_name = os.environ.get('TPU_NAME', 'local')
    
#     print(f"Initializing TPU host {host_id} of {num_hosts}")
    
#     # Initialize distributed JAX
#     jax.distributed.initialize(
#         coordinator_address=f"{tpu_name}:8476",
#         num_processes=num_hosts,
#         process_id=host_id
#     )
    
#     # Verify initialization
#     print(f"JAX process index: {jax.process_index()}")
#     print(f"JAX process count: {jax.process_count()}")
#     print(f"Local devices: {len(jax.local_devices())}")
#     print(f"Global devices: {len(jax.devices())}")
    
#     # Should see 8 local devices and 16 global devices
#     assert len(jax.devices()) == 16, f"Expected 16 devices, got {len(jax.devices())}"
#     print("TPU initialization successful!")

# if __name__ == "__main__":
#     initialize_tpu()
    
#     # Your training code goes here
#     print("Ready for training...")



#!/usr/bin/env python3

import jax
import jax.distributed
import os

def initialize_tpu():
    """Initialize TPU v3-16 multi-host setup"""
    import time
    
    start_time = time.time()
    
    # Get TPU environment info
    host_id = int(os.environ.get('TPU_PROCESS_ID', 0))
    num_hosts = int(os.environ.get('TPU_PROCESS_COUNT', 2))
    tpu_name = os.environ.get('TPU_NAME', 'local')
    
    print(f"Initializing TPU host {host_id} of {num_hosts}")
    print(f"Coordinator: {tpu_name}:8476")
    
    # Initialize distributed JAX
    init_start = time.time()
    jax.distributed.initialize(
        coordinator_address=f"{tpu_name}:8476",
        num_processes=num_hosts,
        process_id=host_id
    )
    print(f"JAX distributed init took: {time.time() - init_start:.2f}s")
    
    # Verify initialization
    print(f"JAX process index: {jax.process_index()}")
    print(f"JAX process count: {jax.process_count()}")
    print(f"Local devices: {len(jax.local_devices())}")
    print(f"Global devices: {len(jax.devices())}")
    
    # Should see 8 local devices and 16 global devices
    assert len(jax.devices()) == 16, f"Expected 16 devices, got {len(jax.devices())}"
    print("TPU initialization successful!")

if __name__ == "__main__":
    initialize_tpu()
    
    # Your training code goes here
    print("Ready for training...")