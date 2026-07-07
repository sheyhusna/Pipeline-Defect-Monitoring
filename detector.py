# """
# detector.py
#
# Loads YOLO model and performs detection.
# """
# import cv2
# from config import MODEL_PATH
#
# import torch
# from ultralytics import YOLO
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# model = YOLO("weights/best.pt")
# model.to(device)
#
# print(f"Using device: {device}")
#
# # Load once when program starts
# model = YOLO(MODEL_PATH)
#
#
# def detect_frame(frame, conf_threshold=0.4):
#
#     results = model(frame, conf=conf_threshold)
#
#     annotated_frame = frame.copy()
#
#     corrosion_found = False
#     max_conf = 0
#
#     for r in results:
#
#         if r.boxes is None:
#             continue
#
#         for box in r.boxes:
#
#             cls = int(box.cls[0])
#             conf = float(box.conf[0])
#
#             label = r.names[cls]
#
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#
#             if label == "corrosion":
#                 color = (0, 0, 255)
#                 corrosion_found = True
#
#                 if conf > max_conf:
#                     max_conf = conf
#             else:
#                 color = (0, 255, 0)
#
#             cv2.rectangle(
#                 annotated_frame,
#                 (x1, y1),
#                 (x2, y2),
#                 color,
#                 4
#             )
#
#             cv2.putText(
#                 annotated_frame,
#                 f"{label} {conf:.2f}",
#                 (x1, y1 - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6,
#                 color,
#                 3
#             )
#
#     return annotated_frame, corrosion_found, max_conf


"""
detector.py

Loads YOLO model and performs corrosion detection.
Uses GPU automatically if CUDA is available.
"""

import cv2
import torch

from ultralytics import YOLO
from config import MODEL_PATH

# =====================================================
# DEVICE SELECTION
# =====================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 50)
print(f"Using device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("=" * 50)

# =====================================================
# LOAD MODEL
# =====================================================

model = YOLO(MODEL_PATH)
model.to(device)

# =====================================================
# DETECTION FUNCTION
# =====================================================

def detect_frame(frame, conf_threshold=0.4):

    results = model(
        frame,
        conf=conf_threshold,
        verbose=False
    )

    annotated_frame = frame.copy()

    corrosion_found = False
    max_conf = 0

    for r in results:

        if r.boxes is None:
            continue

        for box in r.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            label = r.names[cls]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            if label.lower() == "corrosion":

                color = (0, 0, 255)

                corrosion_found = True

                max_conf = max(
                    max_conf,
                    conf
                )

            else:

                color = (0, 255, 0)

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                color,
                4
            )

            cv2.putText(
                annotated_frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                3
            )

    return (
        annotated_frame,
        corrosion_found,
        max_conf
    )