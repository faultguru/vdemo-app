"""YOLO object detection: the model (PyTorch on CPU, or ONNX Runtime on
DirectML with YOLO_DEVICE=directml), and the check of which detected objects
actually move within the motion segments.

Weights live in STORAGE_ROOT/models (YOLO_MODEL_NAME, plus its ONNX export
for DirectML); ultralytics downloads the .pt there on first use.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from blink_watcher.core.runtime import models_dir
from blink_watcher.settings import get_settings
from blink_watcher.vision.hwtrace import trace, traced
from blink_watcher.vision.motion import MotionSegment

_settings = get_settings()

log = logging.getLogger(__name__)

YOLO_MODEL_NAME = "yolov8s.pt"  # small variant - ~22MB, more accurate than nano
YOLO_CONF_THRESHOLD = _settings.yolo_conf_threshold  # min confidence of a detection in the motion check
YOLO_ONNX_IMGSZ = (384, 640)    # directml: static ONNX input (h, w) - the 16:9 letterbox PyTorch uses for 1080p at 640
YOLO_ONNX_BATCH = 1             # directml: static ONNX batch size
MOTION_OVERLAP = _settings.yolo_motion_overlap       # a detection counts if this fraction of its box lies in the motion bbox
MIN_SHIFT_RATIO = _settings.yolo_min_shift_ratio     # ...and its center moves this fraction of the frame's long side across keyframes
# COCO classes that can plausibly move in a residential scene. Everything else
# (tv, chair, potted plant, bench, ...) is static scenery, dropped even when
# YOLO lands a confident box on the motion.
YOLO_MOVING_CLASSES = frozenset({
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "bird", "cat", "dog", "horse", "sheep", "cow",
    "skateboard", "sports ball", "frisbee",
})


# ---------- Model ----------

_model = None


def _weights() -> Path:
    return models_dir(get_settings()) / YOLO_MODEL_NAME


def _load_cpu_yolo():
    from ultralytics import YOLO
    log.info("Loading YOLO model %s (device: cpu)", _weights())
    return YOLO(str(_weights()))


def get_yolo():
    """The model, loaded once per process: ultralytics' predict()/names
    interface on CPU, or on DirectML when YOLO_DEVICE=directml (CPU if that fails)."""
    global _model
    if _model is None:
        if get_settings().yolo_device == "directml":
            try:
                _model = _DirectMLYolo()
            except Exception as e:
                log.warning("DirectML YOLO init failed (%s: %s) - falling back to CPU", type(e).__name__, e)
                trace("yolo.dml.fallback", stage="init", error=f"{type(e).__name__}: {e}"[:500])
        if _model is None:
            _model = _load_cpu_yolo()
    return _model


class _TracedSession:
    """An onnxruntime session whose every run() leaves a crash-safe trace line
    (hwtrace) - after a GPU hang the last line names the call."""

    def __init__(self, session):
        self._session = session
        self.calls = 0

    def run(self, output_names, feeds):
        self.calls += 1
        with traced("yolo.dml.infer", batch=YOLO_ONNX_BATCH, call=self.calls):
            return self._session.run(output_names, feeds)

    def __getattr__(self, name):
        return getattr(self._session, name)


class _DirectMLYolo:
    """YOLO on ONNX Runtime's DirectML provider (any DirectX 12 GPU) with the
    ultralytics model's predict()/names interface.

    ultralytics still does letterboxing, NMS and Results; it only ever opens
    ONNX models on CUDA/CoreML/CPU, so its CPU session is swapped for a
    DirectML one right before the first inference. The model has a static
    batch of YOLO_ONNX_BATCH (fixed shapes are kindest to old GPU drivers), so
    predict() feeds frames in chunks of that size, padding the last one.
    Any DirectML error -> warning + trace, then the CPU model takes over for
    the rest of the process."""

    def __init__(self):
        import onnxruntime as ort
        from ultralytics import YOLO
        from ultralytics.utils import checks

        # ultralytics' ONNX backend pip-installs the "onnxruntime" distribution
        # when missing - over onnxruntime-directml's files. Never auto-install.
        checks.AUTOINSTALL = False
        onnx_path = _ensure_onnx()
        if "DmlExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(f"onnxruntime {ort.__version__} has no DmlExecutionProvider (install onnxruntime-directml)")
        log.info("Loading YOLO model %s (device: directml)", onnx_path)
        with traced("yolo.dml.session_create", model=onnx_path.name):
            session = ort.InferenceSession(str(onnx_path), providers=["DmlExecutionProvider", "CPUExecutionProvider"])
        providers = session.get_providers()
        trace("yolo.dml.providers", providers=providers)
        if providers[0] != "DmlExecutionProvider":
            raise RuntimeError(f"DirectML session fell back to {providers}")
        self._session = _TracedSession(session)
        self._yolo = YOLO(str(onnx_path), task="detect")
        self._yolo.add_callback("on_predict_start", self._use_dml_session)
        self._cpu = None
        # First inference now, so a broken driver falls back at load time rather than mid-scan.
        blank = np.zeros((YOLO_ONNX_IMGSZ[0], YOLO_ONNX_IMGSZ[1], 3), dtype=np.uint8)
        with traced("yolo.dml.first_infer", batch=YOLO_ONNX_BATCH):
            self._predict_dml([blank], verbose=False)
        log.info("YOLO inference on DirectML (%s, batch %d)", ", ".join(providers), YOLO_ONNX_BATCH)

    def _use_dml_session(self, predictor) -> None:
        backend = predictor.model.backend
        if backend.session is not self._session:
            backend.session = self._session

    def _predict_dml(self, source, **kwargs):
        frames = [source] if isinstance(source, np.ndarray) else list(source)
        results = []
        for i in range(0, len(frames), YOLO_ONNX_BATCH):
            chunk = frames[i:i + YOLO_ONNX_BATCH]
            pad = YOLO_ONNX_BATCH - len(chunk)
            results.extend(self._yolo.predict(chunk + [chunk[-1]] * pad, **kwargs)[:len(chunk)])
        return results

    @property
    def names(self) -> dict[int, str]:
        return (self._cpu or self._yolo).names

    def predict(self, source, **kwargs):
        if self._cpu is not None:
            return self._cpu.predict(source, **kwargs)
        try:
            return self._predict_dml(source, **kwargs)
        except Exception as e:
            log.warning("DirectML YOLO inference failed (%s: %s) - using CPU for the rest of this run",
                        type(e).__name__, e)
            trace("yolo.dml.fallback", stage="infer", error=f"{type(e).__name__}: {e}"[:500])
            self._session = self._yolo = None  # release the GPU session
            self._cpu = _load_cpu_yolo()
            return self._cpu.predict(source, **kwargs)


def _ensure_onnx() -> Path:
    """STORAGE_ROOT/models/<name>_<h>x<w>_b<batch>.onnx, exported (on CPU) from the .pt on first use."""
    pt_path = _weights()
    h, w = YOLO_ONNX_IMGSZ
    onnx_path = pt_path.with_name(f"{pt_path.stem}_{h}x{w}_b{YOLO_ONNX_BATCH}.onnx")
    if not onnx_path.exists():
        from ultralytics import YOLO
        log.info("Exporting %s to %s (CPU, one-time)", pt_path.name, onnx_path.name)
        exported = YOLO(str(pt_path)).export(
            format="onnx", imgsz=YOLO_ONNX_IMGSZ, batch=YOLO_ONNX_BATCH, simplify=False, device="cpu", verbose=False,
        )
        Path(exported).replace(onnx_path)
    return onnx_path


# ---------- Moving objects in the motion segments ----------

def _overlap(det_xyxy: list[float], motion_xywh: tuple[int, int, int, int]) -> float:
    """Fraction of a detection's box (x1, y1, x2, y2) inside the motion bbox (x, y, w, h)."""
    dx1, dy1, dx2, dy2 = det_xyxy
    mx, my, mw, mh = motion_xywh
    ix1, iy1 = max(dx1, mx), max(dy1, my)
    ix2, iy2 = min(dx2, mx + mw), min(dy2, my + mh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1) / max(1.0, (dx2 - dx1) * (dy2 - dy1))


def detect_moving_objects(segments: list[MotionSegment], frames: dict[int, np.ndarray],
                          fps: float, info: dict) -> dict[int, list[dict]]:
    """YOLO on each segment's keyframes (`frames`: frame index -> BGR frame):
    {segment index: detections}, keeping only objects that move. A class counts if
      (a) its best detection per keyframe shifts center by >= MIN_SHIFT_RATIO
          of the frame's long side across the keyframes, or
      (b) it is seen in one keyframe only, the segment's first or last
          (it entered or left the scene)."""
    if not segments or not frames:
        return {}
    model = get_yolo()
    names = model.names
    min_shift_px = (max(info.get("width") or 0, info.get("height") or 0) or 1) * MIN_SHIFT_RATIO

    per_segment: dict[int, list[dict]] = {}
    for seg in segments:
        kf_list = [(k, frames[i]) for k, t in enumerate(seg.keyframe_times_sec)
                   if (i := int(round(t * fps))) in frames]
        if not kf_list:
            per_segment[seg.index] = []
            continue
        results = model.predict([f for _, f in kf_list], conf=YOLO_CONF_THRESHOLD, verbose=False)

        # Per class: the best detection per keyframe, (confidence, xyxy).
        best: dict[str, dict[int, tuple[float, list[float]]]] = {}
        for (kf, _), r in zip(kf_list, results):
            if r.boxes is None:
                continue
            for box in r.boxes:
                name = names.get(int(box.cls.item()), str(int(box.cls.item())))
                if name not in YOLO_MOVING_CLASSES:
                    continue
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                if _overlap(xyxy, seg.bbox) < MOTION_OVERLAP:
                    continue
                conf = float(box.conf.item())
                per_kf = best.setdefault(name, {})
                if kf not in per_kf or conf > per_kf[kf][0]:
                    per_kf[kf] = (conf, xyxy)

        kept: list[dict] = []
        last_kf = len(seg.keyframe_times_sec) - 1
        for name, per_kf in best.items():
            centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for _, b in per_kf.values()]
            if len(centers) >= 2:
                xs, ys = [c[0] for c in centers], [c[1] for c in centers]
                shift = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
                moved = shift >= min_shift_px
            else:
                shift = 0.0
                moved = next(iter(per_kf)) in (0, last_kf)
            if not moved:
                continue
            conf, xyxy = max(per_kf.values(), key=lambda v: v[0])
            kept.append({
                "class": name,
                "confidence": round(conf, 3),
                "bbox": [int(v) for v in xyxy],
                "keyframe_shift_px": int(shift),
                "present_in_keyframes": sorted(per_kf),
            })
        per_segment[seg.index] = sorted(kept, key=lambda d: -d["confidence"])
    return per_segment


def yolo_verdict(record: dict, required_classes: Optional[list[str]]) -> dict:
    """Set the yolo processor record's verdict: "passed" if one of
    `required_classes` was detected moving (any class when empty)."""
    found = set(record.get("classes_detected") or [])
    record["required_classes"] = list(required_classes or [])
    record["passed"] = bool(found & set(required_classes)) if required_classes else bool(found)
    return record
