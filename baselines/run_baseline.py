"""
Run the SN-GSR-2027 baseline on every match half of a split.

    calibration -> tracking (+CLIP) -> jersey_verification (Qwen3-VL)
    -> remove_duplicates -> tracklets -> refine -> pitch_projection -> export

Each step runs in its own process (GPU memory is released between steps) and
logs to ``OUTPUT_DIR/<split>/<sequence>/logs/<step>.log``. A full half takes
many hours, so the run is resumable: finished steps are recorded in
``status.json`` and skipped when the command is launched again.

At the end, the per-half predictions are gathered in
``OUTPUT_DIR/<split>/submission.zip`` (one ``<sequence>.json`` per half at the
root of the archive), ready to be scored with ``evaluation/scoring.py``.

Examples:
    python run_baseline.py                                  # every half of SPLIT
    python run_baseline.py --sequences SNGS-0001            # a single half
    python run_baseline.py --from-step refine               # re-run from a step
    python run_baseline.py --video                          # also render an MP4
    python run_baseline.py --set SPLIT=challenge DATA_DIR=/data/SN-GSR-2027
"""

import argparse
import json
import os
import subprocess
import sys
import time
import zipfile

from sngsr.utils import BASELINE_DIR, list_sequences, load_config, resolve, sequence_paths

STEPS = [
    ("calibration", "sngsr.calibration"),
    ("tracking", "sngsr.tracking"),
    ("jersey_verification", "sngsr.jersey.verify"),
    ("remove_duplicates", "sngsr.idatr.remove_duplicates"),
    ("tracklets", "sngsr.idatr.generate_tracklets"),
    ("refine", "sngsr.idatr.refine_tracklets"),
    ("pitch_projection", "sngsr.idatr.project_to_pitch"),
    ("export", "sngsr.export"),
]
STEP_NAMES = [name for name, _ in STEPS]


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run_sequence(config_path, cfg, sequence, from_step=None, video=False, overrides=()):
    paths = sequence_paths(cfg, sequence)
    if not os.path.isdir(paths.img_dir):
        raise FileNotFoundError(f"Frames not found: {paths.img_dir}")
    log_dir = os.path.join(paths.work_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    status_path = os.path.join(paths.work_dir, "status.json")
    status = {}
    if os.path.exists(status_path):
        with open(status_path) as f:
            status = json.load(f)
    if from_step:
        for name in STEP_NAMES[STEP_NAMES.index(from_step):]:
            status.pop(name, None)

    steps = list(STEPS)
    if not cfg.get("QWEN", {}).get("ENABLED", True):
        steps = [s for s in steps if s[0] != "jersey_verification"]
    if video:
        steps.append(("video", "sngsr.visualize"))
        status.pop("video", None)

    for name, module in steps:
        if status.get(name, {}).get("state") == "done":
            log(f"{sequence} | {name}: already done, skipped")
            continue
        cmd = [sys.executable, "-m", module, "--config", config_path, "--sequence", sequence]
        if overrides:
            cmd += ["--set", *overrides]
        log(f"{sequence} | {name}: {' '.join(cmd[1:])}")
        t0 = time.time()
        with open(os.path.join(log_dir, f"{name}.log"), "a") as lf:
            lf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
            lf.flush()
            rc = subprocess.run(cmd, cwd=BASELINE_DIR, stdout=lf, stderr=subprocess.STDOUT).returncode
        dt = round(time.time() - t0)
        status[name] = {"state": "done" if rc == 0 else "failed", "seconds": dt}
        with open(status_path, "w") as f:
            json.dump(status, f, indent=2)
        log(f"{sequence} | {name}: exit={rc} ({dt / 3600:.2f} h)")
        if rc != 0:
            raise RuntimeError(f"Step '{name}' failed on {sequence}, see {log_dir}/{name}.log "
                               "(relaunch the same command to resume from this step)")
    return paths.predictions


def build_submission(cfg, prediction_files):
    out = os.path.join(resolve(cfg["OUTPUT_DIR"]), cfg["SPLIT"], "submission.zip")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in prediction_files:
            zf.write(path, arcname=os.path.basename(path))
    log(f"Submission archive: {out} ({len(prediction_files)} half/halves)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=os.path.join(BASELINE_DIR, "configs", "baseline.yaml"))
    parser.add_argument("--sequences", nargs="+", default=None,
                        help="Sequences (match halves) to process. Default: every one of DATA_DIR/SPLIT")
    parser.add_argument("--from-step", choices=STEP_NAMES, default=None,
                        help="Forget the status of this step and of the following ones")
    parser.add_argument("--video", action="store_true", help="Also render an annotated MP4 per half")
    parser.add_argument("--set", dest="overrides", nargs="+", default=[], metavar="KEY=VALUE",
                        help="Override config entries for every step, e.g. "
                             "--set SPLIT=challenge DATA_DIR=/data/SN-GSR-2027 QWEN.MODEL_ID=Qwen/Qwen3-VL-2B-Instruct")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    cfg = load_config(config_path, args.overrides)
    sequences = args.sequences or list_sequences(cfg)
    log(f"{len(sequences)} sequence(s) in {cfg['SPLIT']}: {', '.join(sequences)}")

    t0 = time.time()
    predictions = []
    for sequence in sequences:
        try:
            predictions.append(run_sequence(config_path, cfg, sequence, args.from_step, args.video, args.overrides))
        except (RuntimeError, FileNotFoundError) as exc:
            log(f"ERROR: {exc}")
    if predictions:
        build_submission(cfg, predictions)
    log(f"Finished in {(time.time() - t0) / 3600:.1f} h")
    if len(predictions) != len(sequences):
        sys.exit(1)


if __name__ == "__main__":
    main()
