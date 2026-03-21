#!/usr/bin/env python3
"""
ExportCollage — unified export for TextCollage YOLO models.

Primary formats (phone app deployment):
    coreml   — iOS via Apple Neural Engine / GPU
    ncnn     — Android / Linux via Vulkan GPU

Secondary formats:
    openvino    — Intel CPU/GPU inference
    torchscript — PyTorch mobile (.ptl)
    executorch  — ExecuTorch .pte (Vulkan or XNNPACK)

Output structure:
    models/<prefix>/
        coreml/         <prefix>.mlpackage
        ncnn/           <prefix>_ncnn_model/
        openvino/       <prefix>.xml  +  <prefix>.bin
        torchscript/    <prefix>.torchscript
        executorch/     <prefix>_<backend>.pte

Usage:
    # Export all formats (default) with a prefix:
    python ExportCollage.py --weights runs/detect/yolo12n/weights/best.pt --prefix TextCollage_v2

    # Export only the two you actually use:
    python ExportCollage.py --weights best.pt --prefix TextCollage_v2 coreml ncnn

    # Prefix defaults to the training run folder name if omitted:
    python ExportCollage.py --weights runs/detect/yolo12n/weights/best.pt
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO


class ExportCollage:
    """Export a trained YOLO model to deployment formats."""

    FORMATS = ("coreml", "ncnn", "openvino", "torchscript", "onnx", "executorch")

    def __init__(
        self,
        weights: str | Path,
        prefix: str | None = None,
        output_dir: str | Path = "models",
        imgsz: int = 640,
    ):
        self.weights = Path(weights).resolve()
        if not self.weights.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights}")

        # Derive prefix from the training run folder if not given.
        # e.g. runs/detect/yolo12n/weights/best.pt  →  "yolo12n"
        #      YOLO11n_pte/YOLO11n_pte_PREP/weights/best.pt  →  "YOLO11n_pte_PREP"
        if prefix is None:
            prefix = self.weights.parent.parent.name
        self.prefix = prefix

        self.output_root = Path(output_dir).resolve() / self.prefix
        self.imgsz = imgsz
        self._model = None

    @property
    def model(self) -> YOLO:
        if self._model is None:
            print(f"Loading model from: {self.weights}")
            self._model = YOLO(str(self.weights))
        return self._model

    # ------------------------------------------------------------------
    # Primary formats
    # ------------------------------------------------------------------

    def to_coreml(self, *, nms: bool = True, half: bool = False) -> Path:
        """Export to Core ML (.mlpackage) for iOS."""
        dest_dir = self.output_root / "coreml"
        dest_dir.mkdir(parents=True, exist_ok=True)

        print(f"Exporting Core ML (imgsz={self.imgsz}, nms={nms}, half={half}) ...")
        raw = Path(self.model.export(
            format="coreml",
            imgsz=self.imgsz,
            nms=nms,
            half=half,
        ))

        dest = dest_dir / f"{self.prefix}.mlpackage"
        _move_replace(raw, dest)

        print(f"  -> {dest}")
        return dest

    def to_ncnn(self, *, half: bool = False, batch: int = 1) -> Path:
        """Export to NCNN for Vulkan GPU inference on Android/Linux."""
        dest_dir = self.output_root / "ncnn"
        dest_dir.mkdir(parents=True, exist_ok=True)

        print(f"Exporting NCNN (imgsz={self.imgsz}, half={half}, batch={batch}) ...")
        raw = Path(self.model.export(
            format="ncnn",
            imgsz=self.imgsz,
            half=half,
            batch=batch,
        ))

        dest = dest_dir / f"{self.prefix}_ncnn_model"
        _move_replace(raw, dest)

        print(f"  -> {dest}")
        return dest

    # ------------------------------------------------------------------
    # Secondary formats
    # ------------------------------------------------------------------

    def to_openvino(self, *, half: bool = False, batch: int = 1) -> Path:
        """Export to OpenVINO IR (.xml + .bin)."""
        dest_dir = self.output_root / "openvino"
        dest_dir.mkdir(parents=True, exist_ok=True)

        print(f"Exporting OpenVINO (imgsz={self.imgsz}, half={half}) ...")
        raw = Path(self.model.export(
            format="openvino",
            imgsz=self.imgsz,
            half=half,
            batch=batch,
        ))

        # Ultralytics returns a directory; move contents with proper names.
        for src_file in raw.iterdir():
            suffix = src_file.suffix
            dest = dest_dir / f"{self.prefix}{suffix}"
            _move_replace(src_file, dest)
            print(f"  -> {dest}")

        # Clean up the now-empty raw directory
        if raw.is_dir():
            shutil.rmtree(raw, ignore_errors=True)

        return dest_dir

    def to_torchscript(self) -> Path:
        """Export to TorchScript."""
        dest_dir = self.output_root / "torchscript"
        dest_dir.mkdir(parents=True, exist_ok=True)

        print(f"Exporting TorchScript (imgsz={self.imgsz}) ...")
        raw = Path(self.model.export(
            format="torchscript",
            imgsz=self.imgsz,
        ))

        dest = dest_dir / f"{self.prefix}.torchscript"
        _move_replace(raw, dest)

        print(f"  -> {dest}")
        return dest

    def to_onnx(self, *, half: bool = False, simplify: bool = True) -> Path:
        """Export to ONNX (.onnx)."""
        dest_dir = self.output_root / "onnx"
        dest_dir.mkdir(parents=True, exist_ok=True)

        print(f"Exporting ONNX (imgsz={self.imgsz}, half={half}, simplify={simplify}) ...")
        raw = Path(self.model.export(
            format="onnx",
            imgsz=self.imgsz,
            half=half,
            simplify=simplify,
        ))

        dest = dest_dir / f"{self.prefix}.onnx"
        _move_replace(raw, dest)

        print(f"  -> {dest}")
        return dest

    def to_executorch(self, *, backend: str = "vulkan") -> Path:
        """
        Export to ExecuTorch .pte.

        Uses the Ultralytics built-in exporter which defaults to XNNPACK.
        For Vulkan, requires ExecuTorch built with Vulkan support.
        """
        import torch
        from torch.export import export as torch_export

        dest_dir = self.output_root / "executorch"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{self.prefix}_{backend}.pte"

        backend = backend.strip().lower()

        print(f"Exporting ExecuTorch (backend={backend}, imgsz={self.imgsz}) ...")

        if backend == "xnnpack":
            # Ultralytics native export handles XNNPACK
            raw = Path(self.model.export(
                format="executorch",
                imgsz=self.imgsz,
            ))
            _move_replace(raw, dest)
        elif backend == "vulkan":
            # Manual lowering through ExecuTorch Vulkan partitioner
            try:
                from executorch.backends.vulkan.partitioner import VulkanPartitioner
                from executorch.exir import to_edge
            except ImportError:
                print("ERROR: ExecuTorch Vulkan backend not available.")
                print("See: https://pytorch.org/executorch/stable/build-run-vulkan.html")
                sys.exit(1)

            inner = self.model.model
            if hasattr(inner, "model") and hasattr(inner.model, "forward"):
                inner = inner.model
            inner = inner.float().eval()

            example = (torch.randn(1, 3, self.imgsz, self.imgsz),)
            exported = torch_export(inner, example)
            edge = to_edge(exported).to_backend(VulkanPartitioner())
            et_program = edge.to_executorch()

            with open(dest, "wb") as f:
                et_program.write_to_file(f)
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'vulkan' or 'xnnpack'.")

        print(f"  -> {dest}")
        return dest

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def export(self, formats: list[str], **kwargs) -> dict[str, Path]:
        """Export to multiple formats. Returns {format_name: output_path}."""
        if "all" in formats:
            formats = list(self.FORMATS)

        dispatch = {
            "coreml": self.to_coreml,
            "ncnn": self.to_ncnn,
            "openvino": self.to_openvino,
            "torchscript": self.to_torchscript,
            "onnx": self.to_onnx,
            "executorch": self.to_executorch,
        }

        results = {}
        for fmt in formats:
            fmt = fmt.strip().lower()
            if fmt not in dispatch:
                print(f"WARNING: Unknown format '{fmt}', skipping. "
                      f"Valid: {', '.join(dispatch)}")
                continue
            results[fmt] = dispatch[fmt](**kwargs)

        print(f"\nDone. Exported {len(results)} format(s) to: {self.output_root}")
        return results


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _move_replace(src: Path, dst: Path) -> None:
    """Move src to dst, replacing dst if it already exists."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    shutil.move(str(src), str(dst))


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export TextCollage YOLO model to deployment formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python ExportCollage.py --weights runs/detect/yolo12n/weights/best.pt --prefix TextCollage_v2
  python ExportCollage.py --weights best.pt --prefix TextCollage_v2 coreml ncnn
  python ExportCollage.py --weights best.pt --imgsz 640 --half ncnn
        """,
    )

    parser.add_argument(
        "formats",
        nargs="*",
        default=["all"],
        help=f"Export format(s): {', '.join(ExportCollage.FORMATS)}, or 'all' (default: all)",
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to .pt weights file",
    )
    parser.add_argument(
        "--prefix", type=str, default=None,
        help="Prefix for output folder and all exported filenames (default: derived from weights path)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="models",
        help="Root output directory (default: models/)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--half", action="store_true",
        help="Use FP16 half precision",
    )
    parser.add_argument(
        "--nms", action="store_true",
        help="Include NMS in Core ML export",
    )
    parser.add_argument(
        "--batch", type=int, default=1,
        help="Batch size for NCNN/OpenVINO export (default: 1)",
    )
    parser.add_argument(
        "--et-backend", type=str, default="vulkan",
        choices=["vulkan", "xnnpack"],
        help="ExecuTorch backend (default: vulkan)",
    )

    args = parser.parse_args()

    exporter = ExportCollage(
        weights=args.weights,
        prefix=args.prefix,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
    )

    # Build per-format kwargs from CLI args
    fmt_kwargs = {}
    if args.half:
        fmt_kwargs["half"] = True
    if args.nms:
        fmt_kwargs["nms"] = True
    if args.batch != 1:
        fmt_kwargs["batch"] = args.batch

    # For executorch, pass backend separately since it's format-specific
    formats = [f.strip().lower() for f in args.formats]

    for fmt in formats if "all" not in formats else list(ExportCollage.FORMATS):
        if fmt == "coreml":
            exporter.to_coreml(
                nms=args.nms,
                half=args.half,
            )
        elif fmt == "ncnn":
            exporter.to_ncnn(
                half=args.half,
                batch=args.batch,
            )
        elif fmt == "openvino":
            exporter.to_openvino(
                half=args.half,
                batch=args.batch,
            )
        elif fmt == "torchscript":
            exporter.to_torchscript()
        elif fmt == "executorch":
            exporter.to_executorch(backend=args.et_backend)
        elif fmt == "all":
            pass  # handled by the outer expansion
        else:
            print(f"WARNING: Unknown format '{fmt}', skipping.")

    print(f"\nAll exports saved under: {exporter.output_root}")


if __name__ == "__main__":
    main()
