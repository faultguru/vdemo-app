"""Video analysis without AI: OpenCV and YOLO.

  video     reading clips: info, frames, JPEG
  ffmpeg    cutting clips
  triage    frame-change triage (is anything happening at all?)
  motion    OpenCV motion segments
  yolo      YOLO model (CPU / DirectML) and moving-object detection
  check     the check pass: triage -> motion -> YOLO
  tracks    YOLO scan + object tracks over a whole clip
  focus     the part of a clip worth keeping, and trimming to it
  sampling  frame sampling for an image-only describer (`frames` command)
  hwtrace   crash-safe trace lines for GPU work
"""
