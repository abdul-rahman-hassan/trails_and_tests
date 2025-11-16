#!/usr/bin/env python3
"""
check_tf_gpu.py

Quick health-check for TensorFlow + GPU availability.
Works with TF 2.x.

Run:
    python check_tf_gpu.py
"""

import sys
import time

def main():
    print("=== TensorFlow GPU check ===\n")
    try:
        import tensorflow as tf
    except Exception as e:
        print("ERROR: Failed to import TensorFlow.")
        print("Exception:", repr(e))
        print("\nSuggestion: Install or activate the environment with TensorFlow (e.g. `pip install tensorflow` or `pip install tensorflow==2.14` depending on your needs).")
        sys.exit(2)

    # Basic TF info
    print("TensorFlow imported successfully.")
    print(f"TF version: {tf.__version__}")

    # Build / runtime info (cuda/cuDNN info availability varies by TF build)
    try:
        build_info = tf.sysconfig.get_build_info()
        # build_info keys depend on TF build; print helpful fields if present
        cuda = build_info.get("cuda_version", None)
        cudnn = build_info.get("cudnn_version", None)
        print("Build info (from tf.sysconfig.get_build_info()):")
        if cuda or cudnn:
            print(f"  cuda_version: {cuda}")
            print(f"  cudnn_version: {cudnn}")
        else:
            # Print the whole dict summary if cuda/cudnn not present
            print("  (cuda / cudnn fields not present in build_info)")
            # optionally show keys to help debugging:
            print("  Available build_info keys:", ", ".join(sorted(build_info.keys())))
    except Exception:
        print("Could not obtain sysconfig build info (this is non-fatal).")

    # Runtime detection APIs
    try:
        # Prefer modern API
        gpus = tf.config.list_physical_devices('GPU')
        cpus = tf.config.list_physical_devices('CPU')
    except Exception as e:
        print("ERROR while listing physical devices:", e)
        gpus = []
        cpus = []

    print(f"\nDetected CPU devices: {len(cpus)}")
    print(f"Detected GPU devices: {len(gpus)}")

    if gpus:
        print("GPUs (physical devices):")
        for i, d in enumerate(gpus):
            print(f"  [{i}] name={d.name}  type={d.device_type}")
        # Try to enable memory growth to avoid OOM on multi-process systems
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
    else:
        print("\nNo GPUs detected by TensorFlow.")
        print("If you expect GPUs, check:")
        print("  * `nvidia-smi` output to ensure the driver sees the GPU")
        print("  * that you installed a TensorFlow build that supports GPU (on many systems it's `pip install tensorflow` with CUDA-enabled variant)")
        print("  * CUDA and cuDNN versions compatible with your TF version")
        print("\nYou can run `nvidia-smi` in terminal (Linux/Windows WSL) to see driver and GPU status.")
        # We'll still continue to run a CPU check

    # Show visible devices (logical)
    try:
        logical = tf.config.list_logical_devices()
        print("\nLogical devices (TF sees):")
        for d in logical:
            print(f"  name={d.name}  type={d.device_type}")
    except Exception:
        pass

    # Small runtime test: create tensors and try to run on GPU device if available
    print("\nRunning small compute test...")

    # Prefer to explicitly test on the first GPU if present
    device_to_test = None
    if gpus:
        device_to_test = gpus[0].name  # e.g. '/physical_device:GPU:0' or '/device:GPU:0'
    else:
        device_to_test = '/CPU:0'

    # Normalize device string to a form usable in tf.device (e.g. '/GPU:0' or '/CPU:0')
    def normalize_device_name(name):
        # typical names: '/physical_device:GPU:0' or '/device:GPU:0' etc
        if name is None:
            return '/CPU:0'
        if 'GPU' in name:
            # extract trailing :N
            idx = name.rfind(':')
            return f"/GPU:{name[idx+1:]}"
        if 'CPU' in name:
            idx = name.rfind(':')
            return f"/CPU:{name[idx+1:]}"
        return name

    test_device = normalize_device_name(device_to_test)
    print(f"Attempting compute on device: {test_device}")

    # Run a small matmul and time it
    import numpy as np
    try:
        with tf.device(test_device):
            # create two moderate matrices that fit easily on GPU
            a = tf.constant(np.random.rand(512, 512).astype(np.float32))
            b = tf.constant(np.random.rand(512, 512).astype(np.float32))
            # warm-up
            c = tf.matmul(a, b)
            # force execution and measure time
            start = time.time()
            for _ in range(3):
                out = tf.matmul(a, b)
            # ensure evaluation (if using eager, this executed already)
            # convert one value to numpy to ensure ops complete
            _ = out[0, 0].numpy()
            duration = time.time() - start
            print(f"Small matmul (3 runs) elapsed time: {duration:.4f} s")
            # report where tensor resides if possible
            try:
                print("Result tensor device:", out.device)
            except Exception:
                pass
            # Validate that device string indicates GPU when expected
            if 'GPU' in out.device.upper():
                print("SUCCESS: operation ran on GPU.")
            else:
                if '/GPU' in test_device:
                    print("WARNING: requested GPU device but operation did not run on GPU.")
                else:
                    print("Operation ran on CPU (expected if no GPU present).")
    except Exception as e:
        print("ERROR during compute test:", repr(e))
        print("This may indicate a problem with CUDA/cuDNN or kernel driver compatibility.")
        print("If you see errors mentioning 'cuDNN' or 'CUDA', check your CUDA/cuDNN versions and TF compatibility.")
        sys.exit(3)

    # Additional helpful checks
    print("\nAdditional checks / tips:")
    # tf.test.is_gpu_available is deprecated; show modern alternatives
    try:
        is_built_with_cuda = tf.test.is_built_with_cuda()
        print(f"tf.test.is_built_with_cuda(): {is_built_with_cuda}")
    except Exception:
        pass

    # tf.test.gpu_device_name() returns GPU device string if available
    try:
        gpu_name = tf.test.gpu_device_name()
        print(f"tf.test.gpu_device_name(): '{gpu_name}'")
    except Exception:
        pass

    print("\nIf you expected GPUs but none were detected, try these:")
    print("  1) Run `nvidia-smi` in your shell to confirm driver and GPU are visible to the OS.")
    print("  2) Ensure the CUDA and cuDNN versions installed match the TensorFlow build. (TF release notes / compatibility matrix).")
    print("  3) Use a TF package that supports GPU - e.g., current TF pip package often includes GPU support (check installation docs for your platform).")
    print("  4) If using Docker, ensure the container has GPU access (nvidia-container-toolkit) and you started container with --gpus flag.")
    print("\nDone.")

if __name__ == "__main__":
    main()
