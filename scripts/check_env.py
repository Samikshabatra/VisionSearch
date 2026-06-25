"""GPU/CUDA smoke test for VisionSearch. Exits 0 only if the RTX 5060 is usable.

Run any time the environment is in doubt:
    .venv/Scripts/python.exe scripts/check_env.py
"""
import sys


def main() -> int:
    import torch

    print(f"torch        : {torch.__version__}")
    print(f"cuda built   : {torch.version.cuda}")
    available = torch.cuda.is_available()
    print(f"cuda runtime : {available}")
    if not available:
        print("FAIL: CUDA not available — install the cu128 build (see requirements notes).")
        return 1

    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"device       : {name} ({total_gb:.1f} GB)")

    # Actually run a tensor op on the GPU — availability alone is not proof.
    x = torch.randn(1024, 1024, device="cuda")
    y = (x @ x).sum().item()
    print(f"matmul on GPU: ok (checksum={y:.1f})")
    print("PASS: GPU is usable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
