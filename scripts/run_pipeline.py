"""One local entry point: environment -> audit -> smoke -> 100 epochs -> four tests -> profile -> report."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import psutil

from utils.experiment import (RunLock, assert_environment, experiment_config, read_json,
                              run_path, save_resolved, training_contract)
from utils.runtime import PROJECT_ROOT, load_checkpoint


def run_pipeline(args):
    cfg = experiment_config(args.config)
    assert_environment(cfg)
    destination = run_path(cfg)
    destination.mkdir(parents=True, exist_ok=True)
    if args.eval_only and not args.checkpoint:
        raise ValueError("--eval-only requires --checkpoint.")
    if args.eval_only and args.resume:
        raise ValueError("--eval-only and --resume are mutually exclusive.")
    if args.checkpoint and not args.eval_only:
        raise ValueError("--checkpoint is for --eval-only; use --resume for training state.")
    # A recorded pre-training OOM adjustment survives a resumed invocation.
    if args.resume:
        saved = load_checkpoint(args.resume)
        previous_cfg = saved["config"]
        if previous_cfg.get("pretraining_adjustments") and cfg["patch_size"] != previous_cfg["patch_size"]:
            cfg["patch_size"] = previous_cfg["patch_size"]
            cfg["pretraining_adjustments"] = previous_cfg["pretraining_adjustments"]
        if saved.get("contract") != training_contract(cfg):
            raise ValueError("Resume configuration differs from the checkpoint contract.")
        del saved
    if not args.resume and not args.eval_only and not args.preflight_only:
        if (destination / "history.csv").exists() or any((destination / "checkpoints").glob("*.pth")):
            raise RuntimeError("Training results already exist. Pass --resume or set a new run_dir; nothing overwritten.")
    status = {"pid": os.getpid(), "state": "starting", "started_unix": time.time(), "completed_stages": [],
              "command": [sys.executable, "-m", "scripts.run_pipeline", *sys.argv[1:]]}

    def write_status(state, **extras):
        status.update(state=state, **extras)
        (destination / "pipeline_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    with RunLock(destination, clear_stale=args.clear_stale_lock) as lock:
        save_resolved(cfg)
        effective_config = destination / "config_resolved.yaml"
        environment = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1", SDD_PIPELINE_PID=str(os.getpid()))

        def stage(name, module, extra=(), allow_failure=False):
            write_status(name)
            command = [sys.executable, "-u", "-m", module, "--config", str(effective_config), *map(str, extra)]
            print(f"STAGE {name}: {subprocess.list2cmdline(command)}", flush=True)
            log_path = destination / ("train.log" if name == "training" else f"{name}.log")
            with log_path.open("a", encoding="utf-8") as log:
                log.write("\nCOMMAND " + subprocess.list2cmdline(command) + "\n")
                log.flush()
                process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
                status["child_pid"] = process.pid
                write_status(name)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                return_code = process.wait()
            status.pop("child_pid", None)
            if return_code:
                if allow_failure:
                    return return_code
                raise subprocess.CalledProcessError(return_code, command)
            status["completed_stages"].append(name)
            write_status(name + "_complete")
            return 0

        try:
            stage("environment", "scripts.check_env")
            stage("data_audit", "scripts.prepare_data")
            if not args.eval_only:
                smoke_code = stage("smoke", "scripts.smoke_test", allow_failure=True)
                while smoke_code:
                    smoke = read_json(destination / "smoke" / "smoke_results.json")
                    reason = smoke.get("reason", "")
                    if args.resume or "out of memory" not in reason.lower() or cfg["patch_size"] <= 128:
                        raise RuntimeError(f"Required smoke checks failed: {reason}")
                    old_patch = cfg["patch_size"]
                    cfg["patch_size"] = 192 if old_patch > 192 else 128
                    cfg.setdefault("pretraining_adjustments", []).append({"old_patch": old_patch,
                        "new_patch": cfg["patch_size"], "reason": reason})
                    print(f"Pre-training OOM adjustment: patch {old_patch} -> {cfg['patch_size']}; recorded in resolved config.", flush=True)
                    save_resolved(cfg)
                    smoke_code = stage("smoke", "scripts.smoke_test", allow_failure=True)
                if args.preflight_only:
                    write_status("preflight_complete", finished_unix=time.time())
                    from utils.reporting import generate_report
                    generate_report(cfg)
                    return
                extra = ["--resume", args.resume] if args.resume else []
                stage("training", "scripts.train_mill", extra)
                state = load_checkpoint(destination / "checkpoints" / "last.pth")
                if state["epoch"] != 100:
                    raise RuntimeError(f"Training returned before epoch 100: {state['epoch']}")
                del state
                checkpoint = destination / "checkpoints" / "best.pth"
            else:
                checkpoint = args.checkpoint
            # Optional metric failures are recorded as NA by evaluate; still profile/report.
            eval_code = stage("evaluation", "scripts.evaluate", ["--checkpoint", checkpoint], allow_failure=True)
            profile_code = stage("profile", "scripts.profile_model", ["--checkpoint", checkpoint], allow_failure=True)
            if profile_code:
                status["profile_failure"] = "Profile process failed; see profile.log (metrics are NA)."
                from utils.evaluation import write_json
                write_json(destination / "metrics" / "model_profile.json", {
                    "parameters": "NA", "parameters_M": "NA", "latency_ms_mean": "NA", "FPS": "NA",
                    "peak_train_allocated_bytes": "NA", "peak_inference_allocated_bytes": "NA",
                    "reason": status["profile_failure"], "input_shape": [1, 3, 400, 600], "precision": "float32"})
            write_status("complete" if eval_code == 0 and profile_code == 0 else "completed_with_evaluation_failures",
                         finished_unix=time.time())
            from utils.reporting import generate_report
            generate_report(cfg)
            if eval_code or profile_code:
                raise RuntimeError("Training state preserved; evaluation/profile failures are recorded in logs/report.")
        except BaseException as error:
            # If the parent is interrupted while a child is still training, keep
            # an exclusive lock owned by that child instead of enabling a duplicate.
            child_pid = status.get("child_pid")
            if child_pid and psutil.pid_exists(child_pid):
                try:
                    child = psutil.Process(child_pid)
                    lock.path.write_text(json.dumps({"pid": child_pid, "created": child.create_time(),
                        "command": "surviving pipeline child", "started_unix": status["started_unix"]}), encoding="utf-8")
                    lock.owned = False
                except psutil.NoSuchProcess:
                    pass
            write_status("interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                         error=f"{type(error).__name__}: {error}", failed_unix=time.time())
            from utils.reporting import generate_report
            generate_report(cfg)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sddnet_v1_100e.yaml")
    parser.add_argument("--resume", help="Resume reliable last.pth to a cumulative total of 100 epochs")
    parser.add_argument("--eval-only", action="store_true", help="Never creates an optimizer or invokes training/smoke")
    parser.add_argument("--checkpoint", help="Evaluation checkpoint, normally checkpoints/best.pth")
    parser.add_argument("--clear-stale-lock", action="store_true", help="Clear only a lock whose PID+creation time is no longer live")
    parser.add_argument("--preflight-only", action="store_true", help="Environment, full data audit and independent smoke checks only")
    args = parser.parse_args()
    if args.preflight_only and args.eval_only:
        parser.error("--preflight-only cannot be combined with --eval-only")
    run_pipeline(args)


if __name__ == "__main__":
    main()
