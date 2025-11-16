#!/usr/bin/env python3
"""
check_torch_gpu.py

Quick health-check for PyTorch + GPU availability.

Run:
    python check_torch_gpu.py
"""
import sys
import time

def main():
    print("=== PyTorch GPU check ===\n")
    try:
        import torch
    except Exception as e:
        print("ERROR: Failed to import PyTorch.")
        print("Exception:", repr(e))
        print("\nSuggestion: install PyTorch in this environment. e.g.:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")  # example; adapt to your CUDA
        sys.exit(2)

    print("PyTorch imported successfully.")
    print("torch.__version__:", torch.__version__)
    # Version info (may be None on CPU-only builds)
    try:
        print("torch.version.cuda:", torch.version.cuda)
    except Exception:
        pass
    try:
        print("torch.backends.cudnn.version():", torch.backends.cudnn.version())
    except Exception:
        pass

    # GPU detection
    try:
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
    except Exception as e:
        print("ERROR querying CUDA:", repr(e))
        cuda_available = False
        device_count = 0

    print(f"\nCUDA available: {cuda_available}")
    print(f"CUDA device count: {device_count}")

    if device_count > 0:
        for i in range(device_count):
            try:
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                total_mem_mb = props.total_memory // (1024 * 1024)
                major = props.major
                minor = props.minor
                print(f"  [{i}] name={name}  compute_capability={major}.{minor}  total_mem={total_mem_mb} MB")
            except Exception as e:
                print(f"  [{i}] (error getting props):", e)
    else:
        print("\nNo CUDA GPUs detected by PyTorch.")

    # logical device / current device info
    if cuda_available and device_count > 0:
        cur = torch.cuda.current_device()
        print(f"\nCurrent CUDA device: {cur} ({torch.cuda.get_device_name(cur)})")
        try:
            print("CUDA device capability (example):", torch.cuda.get_device_properties(cur).major,
                  torch.cuda.get_device_properties(cur).minor)
        except Exception:
            pass

    # Small compute test: matmul on GPU with timing and synchronization
    print("\nRunning small compute test (matmul)...")

    def time_matmul(device_str):
        # Use torch.randn on specified device, run a few matmuls, measure wall time with proper sync
        dev = torch.device(device_str)
        a = torch.randn(512, 512, device=dev, dtype=torch.float32)
        b = torch.randn(512, 512, device=dev, dtype=torch.float32)
        # warm-up
        c = a @ b
        if dev.type == 'cuda':
            torch.cuda.synchronize(dev)
        start = time.time()
        for _ in range(3):
            out = a @ b
        if dev.type == 'cuda':
            torch.cuda.synchronize(dev)
        dur = time.time() - start
        return dur, out

    # choose device to test
    test_device = 'cuda:0' if cuda_available and device_count > 0 else 'cpu'
    print(f"Attempting compute on device: {test_device}")
    try:
        duration, out = time_matmul(test_device)
        print(f"Small matmul (3 runs) elapsed time: {duration:.4f} s")
        # show device info for tensor
        try:
            print("Result tensor device:", out.device)
            if out.device.type == 'cuda':
                print("SUCCESS: operation ran on GPU.")
            else:
                if 'cuda' in test_device:
                    print("WARNING: requested GPU device but operation ran on:", out.device)
                else:
                    print("Operation ran on CPU (expected if no GPU present).")
        except Exception:
            pass
    except Exception as e:
        print("ERROR during compute test:", repr(e))
        if cuda_available:
            print("This may indicate an issue with CUDA/cuDNN or the driver.")
        sys.exit(3)

    # Tiny training step to confirm autograd + optimizer on GPU
    print("\nRunning a tiny training step (one forward/backward + optimizer step)...")
    try:
        device = torch.device(test_device)
        # tiny model
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)
        ).to(device)
        loss_fn = torch.nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        # tiny batch
        x = torch.randn(128, 32, device=device)
        y = torch.randn(128, 10, device=device)

        # forward
        opt.zero_grad()
        preds = model(x)
        loss = loss_fn(preds, y)
        # backward
        loss.backward()
        opt.step()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        print("Tiny training step completed. Loss:", float(loss.item()))
        # confirm that model params are on device
        param_dev = next(model.parameters()).device
        print("Model parameter device:", param_dev)
    except Exception as e:
        print("ERROR during training step:", repr(e))
        sys.exit(4)

    # Extra: print memory usage for CUDA device 0
    if cuda_available and device_count > 0:
        try:
            idx = 0
            torch.cuda.reset_peak_memory_stats(idx)
            # allocate small tensor to register
            _ = torch.randn(1024, 1024, device=f'cuda:{idx}')
            torch.cuda.synchronize(idx)
            peak = torch.cuda.max_memory_allocated(idx)
            print(f"\nCUDA memory (device {idx}) peak allocated: {peak // (1024*1024)} MB")
        except Exception:
            pass

    print("\nAdditional quick checks / tips:")
    print(" - Run `nvidia-smi` to see OS-level GPU visibility, driver version, and running processes.")
    print(" - Verify PyTorch CUDA build: torch.version.cuda (above) shows the CUDA toolchain PyTorch was built for.")
    print(" - If mixed precision desired: try torch.cuda.amp.autocast + GradScaler (works best on Ampere+ GPUs).")
    print("\nDone.")

if __name__ == '__main__':
    main()
