"""
YOLO segmentation is applied to detect the human silhouette only.
The model estimates masks for the person class exclusively; 
no background objects are segmented. This ensures clean inputs for
motion analysis (skeleton, silhouette, optical flow).
"""

import cv2, os
from ultralytics import YOLO

video_path = '/content/****.mp4'

# OpenCV loads the video frame-by-frame
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f'VideoCapture does not open: {video_path}')

orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_path = '/content/****out.mp4'

# Temporary output using mp4v
# (OpenCV often struggles with writing H.264 directly, so mp4v is safer)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, orig_fps, (width, height))
if not out.isOpened():
    raise RuntimeError('VideoWriter does not open')

# ====== YOLO models ======
det_model = YOLO('yolo11s.pt')       # keypoint detection
seg_model = YOLO('yolo11s-seg.pt')   # human segmentation

# Frame-by-frame processing
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === Perform YOLO segmentation ===
    results = seg_model(frame)

    # Visualization — draw the segmentation mask onto the frame
    vis = results[0].plot()

    # Append the frame to the output video
    out.write(vis)
    frame_idx += 1

cap.release()
out.release()
print(f"Segmentation visualization video saved: {out_path}")

# ===== Re-encode with ffmpeg for QuickTime compatibility =====
out_qt_path = '/content/****_qt_seg.mp4'

os.system(
    f"ffmpeg -y -i {out_path} -vcodec libx264 -pix_fmt yuv420p "
    f"-profile:v baseline -level 3.0 -movflags +faststart {out_qt_path}"
)

print("QuickTime-compatible file:", out_qt_path)
