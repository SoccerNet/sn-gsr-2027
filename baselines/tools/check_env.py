"""Check that this machine can run the baseline, before launching hours of compute.

    python tools/check_env.py                      # with configs/baseline.yaml
    python tools/check_env.py --set SPLIT=challenge

Reports the Python packages, the GPU (memory, BF16 / FP8 support), the
checkpoints, the Hugging Face login, the data and the free disk space, and
says which settings to change when something is missing. Exits with status 1
if a blocking problem is found.
"""

import argparse
import importlib
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

OK, WARN, FAIL = "ok  ", "warn", "FAIL"
problems = []


def report(level, name, detail=""):
    print(f"[{level}] {name}{': ' + detail if detail else ''}")
    if level == FAIL:
        problems.append(name)


def check_packages():
    print("\n== Python packages")
    if sys.version_info < (3, 10):
        report(FAIL, "python", f"{sys.version.split()[0]} (>= 3.10 required)")
    else:
        report(OK, "python", sys.version.split()[0])
    for module, pip_name in [
        ("torch", "torch"), ("torchvision", "torchvision"), ("numpy", "numpy<2"), ("cv2", "opencv-python"),
        ("pandas", "pandas"), ("scipy", "scipy"), ("sklearn", "scikit-learn"), ("yaml", "pyyaml"),
        ("albumentations", "albumentations"), ("cython_bbox", "cython_bbox"), ("lap", "lapx"),
        ("clip", "CLIP (git)"), ("torchreid", "torchreid (git)"), ("transformers", "transformers>=4.57"),
        ("accelerate", "accelerate"), ("huggingface_hub", "huggingface_hub"),
        ("yolox", "pip install -e ."), ("sngsr", "pip install -e ."),
    ]:
        try:
            mod = importlib.import_module(module)
            report(OK, module, getattr(mod, "__version__", ""))
        except Exception as exc:  # noqa: BLE001 - any import failure is a finding
            report(FAIL, module, f"{type(exc).__name__}: {exc} (install {pip_name})")
    try:
        import numpy
        if int(numpy.__version__.split(".")[0]) >= 2:
            report(FAIL, "numpy", f"{numpy.__version__}: cython_bbox / torchreid need numpy<2")
    except ImportError:
        pass
    try:
        import transformers
        major, minor = (int(v) for v in transformers.__version__.split(".")[:2])
        if (major, minor) < (4, 57):
            report(FAIL, "transformers", f"{transformers.__version__}: Qwen3-VL needs >= 4.57")
    except (ImportError, ValueError):
        pass


def check_gpu(cfg):
    print("\n== Device")
    try:
        import torch
    except ImportError:
        return
    from sngsr.jersey.qwen import resolve_model_id
    from sngsr.utils import get_device

    device = get_device(cfg)
    report(OK, "torch", f"{torch.__version__}, CUDA {torch.version.cuda}, device {device}")
    if device.type != "cuda":
        report(WARN, "GPU", "no CUDA GPU: every step runs on the CPU, a half would take days")
        return
    props = torch.cuda.get_device_properties(device)
    mem = props.total_memory / 1024**3
    cc = f"{props.major}.{props.minor}"
    report(OK, "GPU", f"{props.name}, {mem:.0f} GB, compute capability {cc}")
    try:
        torch.zeros(1, device=device).sum().item()
        report(OK, "CUDA kernels", "a tensor op ran on the GPU")
    except Exception as exc:  # noqa: BLE001
        report(FAIL, "CUDA kernels", f"{exc} (PyTorch build does not match this GPU / driver)")
    report(OK if torch.cuda.is_bf16_supported() else WARN, "BF16",
           "supported" if torch.cuda.is_bf16_supported() else "not supported, Qwen runs in FP16")

    qcfg = cfg.get("QWEN", {})
    if qcfg.get("ENABLED", True):
        requested = qcfg.get("MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct-FP8")
        model_id = resolve_model_id(requested, device)
        need = 10 if "8B" in model_id and "FP8" in model_id.upper() else 18 if "8B" in model_id else 6
        level = OK if mem >= need + 4 else WARN
        report(level, "Qwen3-VL", f"{model_id} needs ~{need} GB of weights"
               + ("" if level == OK else
                  "; set QWEN.MODEL_ID=Qwen/Qwen3-VL-2B-Instruct or QWEN.ENABLED=false")
               + ("" if model_id == requested else f" (FP8 unsupported, BF16 fallback of {requested})"))


def check_files(cfg):
    from sngsr.utils import list_sequences, resolve

    print("\n== Checkpoints")
    for key in [("SFR", "MODEL_PATH"), ("TRACKER", "MODEL_PATH"), ("REID", "MODEL_PATH"), ("CLIP", "MODEL_PATH")]:
        path = resolve(cfg[key[0]][key[1]])
        if os.path.isfile(path):
            report(OK, ".".join(key), path)
        else:
            report(FAIL, ".".join(key), f"missing {path} (python tools/download_weights.py)")
    clip_dir = os.environ.get("CLIP_HOME") or os.path.expanduser("~/.cache/clip")
    report(OK if os.path.isfile(os.path.join(clip_dir, "ViT-L-14.pt")) else WARN, "CLIP ViT-L/14",
           f"cached in {clip_dir}" if os.path.isfile(os.path.join(clip_dir, "ViT-L-14.pt"))
           else f"not cached yet, ~900 MB will be downloaded to {clip_dir}")

    print("\n== Hugging Face")
    try:
        from huggingface_hub import get_token
        report(OK if get_token() else WARN, "login",
               "token found" if get_token() else "no token: run `hf auth login` (the dataset is gated)")
    except ImportError:
        pass

    print("\n== Data")
    try:
        sequences = list_sequences(cfg)
        split_dir = os.path.join(resolve(cfg["DATA_DIR"]), cfg["SPLIT"])
        report(OK if sequences else FAIL, "sequences",
               f"{len(sequences)} in {split_dir}" if sequences else f"no <sequence>/img1 in {split_dir}")
    except FileNotFoundError as exc:
        report(FAIL, "sequences", f"{exc} (see ../data/README.md, or --set DATA_DIR=... SPLIT=...)")

    out_dir = resolve(cfg["OUTPUT_DIR"])
    probe = out_dir if os.path.isdir(out_dir) else os.path.dirname(out_dir)
    free = shutil.disk_usage(probe).free / 1024**3
    report(OK if free >= 15 else WARN, "free disk for outputs", f"{free:.0f} GB in {probe} (~10 GB per half)")
    report(OK if shutil.which("ffmpeg") else WARN, "ffmpeg",
           "found" if shutil.which("ffmpeg") else "missing (only needed for --video and extract_frames.py)")


def main():
    from sngsr.utils import BASELINE_DIR, load_config

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=os.path.join(BASELINE_DIR, "configs", "baseline.yaml"))
    parser.add_argument("--set", dest="overrides", nargs="+", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)

    check_packages()
    check_gpu(cfg)
    check_files(cfg)
    print()
    if problems:
        print(f"{len(problems)} blocking problem(s): {', '.join(problems)}")
        sys.exit(1)
    print("Ready to run the baseline.")


if __name__ == "__main__":
    main()
