#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# train_linux_plant_all_training_data.py
#
# CUDA 12.1 ===> pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.6 ===> pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
#
# REQUIRED MULTI-GPU LAUNCH COMMAND (NCCL stability on this machine):
#   export NCCL_P2P_DISABLE=1
#   export NCCL_IB_DISABLE=1
#   export NCCL_ASYNC_ERROR_HANDLING=1
#   export NCCL_DEBUG=INFO
#   YOLO_DEVICE='0,1' \
#   /home/brlab/Dropbox/TextCollage/.venv_TC_linux/bin/python \
#   /home/brlab/Dropbox/TextCollage/train_linux_plant_all_training_data.py

"""
MULTI-GPU TRAINING (IMPORTANT – REQUIRED ON THIS MACHINE)

This training script uses PyTorch DDP via Ultralytics. On this system (RTX 6000 Ada,
single host, multiple GPUs), NCCL Peer-to-Peer (P2P) and InfiniBand (IB) transport
can silently deadlock after AMP initialization, causing training to hang at
"AMP: checks passed".

To ensure reliable startup, P2P and IB MUST be disabled and NCCL error handling
enabled. This routes inter-GPU communication through host memory, which is slightly
slower but stable.

REQUIRED LAUNCH COMMAND:

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
YOLO_DEVICE='0,1' \
/home/brlab/Dropbox/TextCollage/.venv_TC_linux/bin/python \
/home/brlab/Dropbox/TextCollage/train_linux_plant_all_training_data.py

Notes:
- The hang is NOT a single-GPU fallback or misconfiguration.
- DDP is correctly launched with 2 processes; the issue is NCCL transport.
- If these variables are not set, training may stall indefinitely after AMP checks.
- Single-GPU runs do not require these flags.

References:
- PyTorch NCCL troubleshooting
- Ultralytics DDP multi-GPU behavior
"""

from ultralytics import YOLO
import os

# W&B live logging via callbacks
import wandb

from train_utils import evaluate_and_log
from export_models_all_formats import export_all_formats

# -----------------------------
# NEW DATASET INTERFACE (minimal changes)
# -----------------------------
DATA_YAML = "/datab/PLANT_detector_data/YOLO/data.yaml"

# Keep your naming “oddities” but swap in dataset-specific names.
WANDB_PROJECT = "LM3_BBOXES_PLANT_YOLO12"
RUN_PREFIX = "plant_alldata_yolo12n"  # used for Ultralytics run names + export naming

# -----------------------------
# SELECTOR
# -----------------------------
# Set to: "n" (nano), "x" (xlarge), or "both"
TRAIN_VARIANT = "n"

# -----------------------------
# MULTI-GPU CONFIG
# -----------------------------
# Use both GPUs for DDP training.
DEVICE = os.environ.get("YOLO_DEVICE", "0,1")

# Batch sizing (global batch). Must be a multiple of GPU count (2).
BATCH_N_640 = 64
BATCH_X_1280 = 8

# Image sizes:
IMGSZ_N = 640
IMGSZ_X = 1280

EPOCHS_N = 500
EPOCHS_X = 500

# -----------------------------
# EXPORT CONFIG (IMPORTANT)
# -----------------------------
# Export MUST be single-device. If you pass "0,1" to export you can hit AutoBatch restrictions.
EXPORT_DEVICE = os.environ.get("YOLO_EXPORT_DEVICE", "0")  # "0" or "cpu"
EXPORT_BATCH = int(os.environ.get("YOLO_EXPORT_BATCH", "1"))

# -----------------------------
# W&B LIVE LOGGING CONFIG
# -----------------------------
WANDB_LOG_EVERY_N = int(os.environ.get("WANDB_LOG_EVERY_N", "10"))


def _wandb_init_if_needed(run_name: str):
    """
    Ultralytics often initializes W&B itself; if it hasn't, we init here.
    This ensures our batch-level callbacks have an active run to log to.
    """
    if wandb.run is None:
        wandb.init(project=WANDB_PROJECT, name=run_name)


def _log_optimizer_lrs(trainer, data: dict):
    opt = getattr(trainer, "optimizer", None)
    if opt is None:
        return
    for i, pg in enumerate(opt.param_groups):
        lr = pg.get("lr", None)
        if lr is not None:
            data[f"lr/pg{i}"] = float(lr)


def _on_train_batch_end(trainer):
    """
    Stream loss/LR updates during training, not just at epoch boundaries.
    """
    step = getattr(trainer, "iter", None)
    if step is None or step % WANDB_LOG_EVERY_N != 0:
        return

    data = {
        "train/epoch": float(getattr(trainer, "epoch", 0)),
        "train/iter": float(step),
    }

    # Best-effort loss extraction (Ultralytics versions differ).
    tloss = getattr(trainer, "tloss", None)
    if tloss is not None:
        try:
            data["train/loss"] = float(tloss)
        except Exception:
            pass

    loss_items = getattr(trainer, "loss_items", None)
    if loss_items is not None:
        try:
            for i, v in enumerate(loss_items):
                data[f"train/loss_{i}"] = float(v)
        except Exception:
            pass

    _log_optimizer_lrs(trainer, data)
    wandb.log(data, step=int(step))


def _on_val_batch_end(trainer):
    """
    Heartbeat logs during validation so W&B visibly updates while val runs.
    """
    validator = getattr(trainer, "validator", None)
    if validator is None:
        return

    vbi = getattr(validator, "batch_i", None)
    step = getattr(trainer, "iter", None)
    if vbi is None or step is None:
        return

    if int(vbi) % WANDB_LOG_EVERY_N != 0:
        return

    wandb.log(
        {
            "val/epoch": float(getattr(trainer, "epoch", 0)),
            "val/batch_i": float(vbi),
        },
        step=int(step),
    )


def _attach_wandb_callbacks(model: YOLO, run_name: str) -> None:
    _wandb_init_if_needed(run_name)
    model.add_callback("on_train_batch_end", _on_train_batch_end)
    model.add_callback("on_val_batch_end", _on_val_batch_end)


if __name__ == '__main__':
    # --- W&B environment ---
    # (keeps Ultralytics integration, but ensures "live" behavior and avoids heavy uploads)
    os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", "online")       # "online" or "offline"
    os.environ["WANDB_PROJECT"] = os.environ.get("WANDB_PROJECT", WANDB_PROJECT)
    os.environ["WANDB_WATCH"] = os.environ.get("WANDB_WATCH", "false")     # avoid heavy gradient watching
    os.environ["WANDB_LOG_MODEL"] = os.environ.get("WANDB_LOG_MODEL", "false")

    trained_models = {}  # name -> best.pt path

    # -----------------------------
    # Train YOLOv12n
    # -----------------------------
    if TRAIN_VARIANT in ("n", "both"):
        run_name = f"{RUN_PREFIX}_yolo12n"
        os.environ["WANDB_NAME"] = run_name

        model = YOLO('yolov12/yolov12n.pt')
        _attach_wandb_callbacks(model, run_name)

        _ = model.train(
            data=DATA_YAML,
            epochs=EPOCHS_N,
            batch=BATCH_N_640,     # must be multiple of GPU count (2)
            workers=12,
            imgsz=IMGSZ_N,
            device=DEVICE,         # DDP: "0,1"
            name=run_name,
            val=True,
            patience=100,

            # memory safety / reproducibility
            cache=False,
            deterministic=True,

            # wandb integration
            exist_ok=True,
        )

        best_n = f"runs/detect/{run_name}/weights/best.pt"

        # Export must be single-device; force batch>=1 for export
        export_all_formats(
            weights=best_n,
            model_name=run_name,
            version="v1.0.0",
            data_yaml=DATA_YAML,
            imgsz=IMGSZ_N,
            device=EXPORT_DEVICE,   # "0" or "cpu"
            batch=EXPORT_BATCH,     # typically 1
        )

        print("Training finished successfully.")
        trained_models["yolo12n"] = best_n

    # -----------------------------
    # Train YOLOv12x (at 1280)
    # -----------------------------
    if TRAIN_VARIANT in ("x", "both"):
        run_name = f"{RUN_PREFIX}_yolo12x"
        os.environ["WANDB_NAME"] = run_name

        modelx = YOLO('yolov12/yolov12x.pt')
        _attach_wandb_callbacks(modelx, run_name)

        _ = modelx.train(
            data=DATA_YAML,
            epochs=EPOCHS_X,
            batch=BATCH_X_1280,    # must be multiple of GPU count (2)
            workers=12,
            imgsz=IMGSZ_X,
            device=DEVICE,
            name=run_name,
            patience=100,

            # (kept your “oddities”: keep x block lean)
        )

        best_x = f"runs/detect/{run_name}/weights/best.pt"

        export_all_formats(
            weights=best_x,
            model_name=run_name,
            version="v1.0.0",
            data_yaml=DATA_YAML,
            imgsz=IMGSZ_X,
            device=EXPORT_DEVICE,
            batch=EXPORT_BATCH,

            # Optional flags retained from your prior pattern
            val=True,
            cache=False,
            deterministic=True,
            exist_ok=True,
        )

        print("Training finished successfully.")
        trained_models["yolo12x"] = best_x

    # -----------------------------
    # Evaluate + log (only what we trained)
    # -----------------------------
    if trained_models:
        data_yamls = {
            "Plant": DATA_YAML,
        }

        evaluate_and_log(
            models=trained_models,
            data_yamls=data_yamls,
            device=DEVICE,
            imgsz=IMGSZ_X if ("yolo12x" in trained_models and "yolo12n" not in trained_models) else IMGSZ_N,
            out_dir="/datab/Component_Detectors_Training/Training_Stats/eval",
            wandb_project="LM2-YOLO12-Evals",
            extra_config={"host": "ubuntu22", "notes": "test-split evaluation"},
        )
