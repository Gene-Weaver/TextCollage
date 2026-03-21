#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Dict, Union
from ultralytics import YOLO
import numpy as np

# --- Optional W&B logging ---
try:
    import wandb
    _WANDB_OK = True
except Exception:
    _WANDB_OK = False


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _extract_metrics(res) -> Dict[str, float]:
    """
    Normalize Ultralytics val() outputs to a flat dict.
    Supports both old and new versions of YOLOv8/YOLOv12.
    """
    out = {}

    d = getattr(res, "results_dict", None)
    if isinstance(d, dict) and d:
        key_map = {
            "precision": ["metrics/precision(B)", "metrics/precision"],
            "recall": ["metrics/recall(B)", "metrics/recall"],
            "map50": ["metrics/mAP50(B)", "metrics/mAP50"],
            "map5095": ["metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP(B)"],
            "fitness": ["fitness"],
            "speed_img": ["speed/img"],
            "speed_nms": ["speed/NMS"],
            "speed_infer": ["speed/inference", "speed/infer"],
        }
        for k, candidates in key_map.items():
            for c in candidates:
                if c in d:
                    out[k] = _to_float(d[c])
                    break

    m = getattr(res, "metrics", None)
    if isinstance(m, dict):
        out.setdefault("precision", _to_float(m.get("precision")))
        out.setdefault("recall", _to_float(m.get("recall")))
        out.setdefault("map50", _to_float(m.get("map50")))
        out.setdefault("map5095", _to_float(m.get("map")))
    else:
        box = getattr(res, "box", None)
        if box is not None:
            out.setdefault("map50", _to_float(getattr(box, "map50", None)))
            out.setdefault("map5095", _to_float(getattr(box, "map", None)))
        out.setdefault("precision", _to_float(getattr(m, "p", None) if m else None))
        out.setdefault("recall", _to_float(getattr(m, "r", None) if m else None))

    return {k: v for k, v in out.items() if v is not None}


def _write_txt_report(out_dir: Path, dataset_name: str, model_name: str, metrics: Dict[str, float]):
    _ensure_dir(out_dir / dataset_name)
    p = out_dir / dataset_name / f"{model_name}_test_metrics.txt"
    lines = [f"{k}: {metrics[k]:.6f}" for k in sorted(metrics.keys())]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"📝 Saved: {p}")


def _log_wandb(project: str, run_name: str, config: dict, metrics: Dict[str, float], artifact_path: str | None = None):
    """Log metrics + optional artifact to W&B."""
    if not _WANDB_OK:
        print("⚠️ wandb not installed; skipping W&B logging.")
        return

    try:
        run = wandb.init(project=project, name=run_name, config=config, reinit=True)
        wandb.log(metrics)
        for k, v in metrics.items():
            wandb.run.summary[k] = v

        if artifact_path and Path(artifact_path).exists():
            art = wandb.Artifact(name=f"{run_name}_weights", type="model")
            art.add_file(artifact_path)
            wandb.log_artifact(art)

        wandb.finish()
        print(f"✅ Logged to Weights & Biases: {project}/{run_name}")
    except Exception as e:
        print(f"⚠️ Failed to log to W&B: {e}")


def evaluate_and_log(
    models: Dict[str, Union[str, YOLO]],
    data_yamls: Dict[str, str],
    *,
    device: str = "0",
    imgsz: int = 640,
    out_dir: str = "/datab/Component_Detectors_Training/Training_Stats/eval",
    wandb_project: str = "YOLO12-Evals",
    extra_config: dict | None = None,
):
    """
    Evaluate multiple YOLO models on multiple datasets, save .txt reports,
    and (optionally) log to Weights & Biases.

    Args:
        models: dict of {model_name: YOLO_or_weight_path}
        data_yamls: dict of {dataset_name: path_to_yaml}
        device: CUDA device string (e.g. "0", "0,1", "cpu")
        imgsz: evaluation image size
        out_dir: where to save .txt metrics
        wandb_project: name of W&B project
        extra_config: optional metadata added to W&B run config
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    for ds_name, yaml_path in data_yamls.items():
        for model_name, mod in models.items():
            model = mod if isinstance(mod, YOLO) else YOLO(mod)
            print(f"\n[Eval] dataset={ds_name}  model={model_name}  split=test")

            res = model.val(
                data=yaml_path,
                split="test",
                imgsz=imgsz,
                device=device,
                verbose=False,
            )

            metrics = _extract_metrics(res)
            if not metrics:
                print("⚠️ No metrics extracted; check Ultralytics version for val() outputs.")

            _write_txt_report(out_dir, ds_name, model_name, metrics)

            run_cfg = {"dataset": ds_name, "yaml": yaml_path, "model": model_name,
                       "imgsz": imgsz, "device": device}
            if extra_config:
                run_cfg.update(extra_config)

            # Optionally log to W&B
            weight_path = getattr(model, "ckpt_path", None)
            _log_wandb(
                wandb_project,
                f"{ds_name}__{model_name}__test",
                run_cfg,
                metrics,
                artifact_path=weight_path
            )

def _safe_loss_items(trainer):
    """
    Return a dict of loss scalars across YOLOv8/YOLOv12 variants.
    Tries label_loss_items() first, then falls back to loss_items tensor/list.
    """
    # preferred path on newer versions
    if hasattr(trainer, "label_loss_items"):
        try:
            return trainer.label_loss_items(trainer.tloss, prefix="loss/")
        except Exception:
            pass

    # fallback: best-effort names
    li = getattr(trainer, "loss_items", None)
    if li is None:
        tl = getattr(trainer, "tloss", None)
        if tl is not None:
            li = tl
    if li is None:
        return {}

    try:
        vals = np.array(li, dtype=float).ravel().tolist()
    except Exception:
        return {}

    names_guess = ["loss/box_loss", "loss/cls_loss", "loss/dfl_loss", "loss/pose_loss"]
    out = {}
    for idx, v in enumerate(vals):
        key = names_guess[idx] if idx < len(names_guess) else f"loss/loss_{idx}"
        out[key] = float(v)
    return out


def log_batch(trainer, LOG_EVERY_N: int = 100, LOG_IMAGES_EVERY_M: int = 1000):
    # ---- robust iteration index ----
    i = getattr(trainer, "batch_i", None)
    if i is None:
        i = getattr(trainer, "iter", None)
    if i is None:
        i = getattr(trainer, "iteration", None)
    if i is None:
        # maintain our own counter on the trainer object
        i = getattr(trainer, "_cb_i", -1) + 1
        setattr(trainer, "_cb_i", i)

    # batches per epoch if available
    nb = getattr(trainer, "nb", None)
    epoch = int(getattr(trainer, "epoch", 0))
    if isinstance(nb, int):
        step = epoch * nb + int(i)
    else:
        bs = int(getattr(trainer, "batch_size", 1)) or 1
        seen = int(getattr(trainer, "seen", 0))
        step = seen // bs

    if i % LOG_EVERY_N == 0:
        metrics = _safe_loss_items(trainer)

        # learning rate across versions
        lr_val = None
        lr_attr = getattr(trainer, "lr", None)
        if isinstance(lr_attr, dict) and lr_attr:
            lr_val = list(lr_attr.values())[0]
        elif hasattr(trainer, "optimizer") and trainer.optimizer and trainer.optimizer.param_groups:
            try:
                lr_val = trainer.optimizer.param_groups[0].get("lr", None)
            except Exception:
                pass
        if lr_val is not None:
            metrics["train/lr"] = float(lr_val)

        # global/iter signals that match your define_metric()
        metrics["train/iter"] = int(i)
        metrics["global/epoch"] = epoch

        wandb.log(metrics, step=step)

    if LOG_IMAGES_EVERY_M and i % LOG_IMAGES_EVERY_M == 0:
        # lightweight sample image logging (best-effort)
        try:
            imgs = trainer.batch[0]  # [B,C,H,W]
            img0 = imgs[0].detach().float().cpu()
            arr = (img0.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype("uint8")
            wandb.log({"train/sample": wandb.Image(arr), "train/iter": int(i)}, step=step)
        except Exception:
            pass