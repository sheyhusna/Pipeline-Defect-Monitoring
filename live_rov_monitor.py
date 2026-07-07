"""
live_rov_monitor.py

BlueROV2 Real-Time Corrosion Monitoring System

Purpose:
- Connect to BlueROV2 RTSP stream
- Run YOLO corrosion detection
- Display detections with bounding boxes
- Show corrosion alerts
- Save corrosion evidence frames
- Optimized for lower latency

Press Q to quit.
"""

import cv2
import os
import time
from datetime import datetime

from detector import detect_frame
from config import RTSP_URL, SAVE_FOLDER


# =====================================================
# SETTINGS
# =====================================================

CONFIDENCE_THRESHOLD = 0.40

# Process every Nth frame
FRAME_SKIP = 5

# Detection input resolution
DETECTION_WIDTH = 640
DETECTION_HEIGHT = 384

# Display resolution
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

# Save evidence image every N processed frames
SAVE_INTERVAL = 30


# =====================================================
# CREATE OUTPUT FOLDER
# =====================================================

os.makedirs(SAVE_FOLDER, exist_ok=True)

print("=" * 50)
print("BlueROV2 Live Monitoring")
print("=" * 50)
print("Connecting to RTSP stream...")

# =====================================================
# CONNECT TO RTSP
# =====================================================

cap = cv2.VideoCapture(RTSP_URL)

# Reduce OpenCV buffering
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("ERROR: Unable to connect to RTSP stream.")
    exit()

print("Connected successfully.")

# =====================================================
# VARIABLES
# =====================================================

frame_id = 0
processed_frames = 0

start_time = time.time()

# =====================================================
# MAIN LOOP
# =====================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("RTSP stream disconnected.")
        break

    frame_id += 1

    # =============================================
    # SKIP FRAMES
    # =============================================

    if frame_id % FRAME_SKIP != 0:
        continue

    processed_frames += 1

    # =============================================
    # RESIZE FOR FASTER DETECTION
    # =============================================

    frame = cv2.resize(
        frame,
        (DETECTION_WIDTH, DETECTION_HEIGHT)
    )

    # =============================================
    # YOLO DETECTION
    # =============================================

    annotated, corrosion, conf = detect_frame(
        frame,
        CONFIDENCE_THRESHOLD
    )

    # =============================================
    # CALCULATE FPS
    # =============================================

    elapsed_time = time.time() - start_time

    if elapsed_time > 0:
        fps = processed_frames / elapsed_time
    else:
        fps = 0

    # =============================================
    # STATUS DISPLAY
    # =============================================

    if corrosion:

        status_text = f"CORROSION DETECTED ({conf:.2f})"
        status_color = (0, 0, 255)

        # Save evidence periodically
        if processed_frames % SAVE_INTERVAL == 0:

            filename = os.path.join(
                SAVE_FOLDER,
                datetime.now().strftime(
                    "live_corrosion_%Y%m%d_%H%M%S.jpg"
                )
            )

            cv2.imwrite(
                filename,
                annotated
            )

            print(f"Evidence saved: {filename}")

    else:

        status_text = "PIPELINE CLEAR"
        status_color = (0, 255, 0)

    # =============================================
    # OVERLAYS
    # =============================================

    cv2.putText(
        annotated,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        status_color,
        2
    )

    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        1
    )

    cv2.putText(
        annotated,
        f"Frames: {processed_frames}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        1
    )

    # =============================================
    # DISPLAY WINDOW
    # =============================================

    display = cv2.resize(
        annotated,
        (DISPLAY_WIDTH, DISPLAY_HEIGHT)
    )

    cv2.imshow(
        "BlueROV2 Real-Time Corrosion Monitoring",
        display
    )

    # =============================================
    # PRESS Q TO QUIT
    # =============================================

    key = cv2.waitKey(1)

    if key == ord("q"):
        print("Monitoring stopped by user.")
        break

# =====================================================
# CLEANUP
# =====================================================

cap.release()
cv2.destroyAllWindows()

print("Monitoring session ended.")