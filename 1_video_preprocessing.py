"""
This script performs video preprocessing only: frame-rate downsampling and
re-encoding. No human detection, segmentation, or keypoint estimation is done
at this stage. The goal is to generate a stable, lightweight, and
QuickTime-compatible video before running skeletal or silhouette analysis.
"""

import cv2, os, numpy as np

# ====== Input video ======
video_path = '/content/*****.mp4'
if not os.path.exists(video_path):
    raise FileNotFoundError(f'Video not found: {video_path}')

# ====== Open video ======
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError('Failed to open VideoCapture.')

ret, test_frame = cap.read()
if not ret:
    raise RuntimeError('Could not read the first frame.')
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
height, width = test_frame.shape[:2]

print(f"fps={orig_fps:.3f}, frames={frame_count}, size={width}x{height}")

# ====== Output settings ======
target_fps = 30.0

out_path_tmp = '/content/****.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(out_path_tmp, fourcc, target_fps, (width, height))

if not out.isOpened():
    raise RuntimeError('Failed to open VideoWriter.')

# ====== Frame skipping loop ======
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % frame_interval == 0:
        out.write(frame)
    frame_idx += 1

cap.release()
out.release()

print("Temporary mp4v file saved:", out_path_tmp)

# ====== Re-encode to QuickTime-compatible format ======
out_path_qt = '/content/****.mp4'
os.system(
    f"ffmpeg -y -i {out_path_tmp} -vcodec libx264 -pix_fmt yuv420p "
    f"-profile:v baseline -level 3.0 -movflags +faststart {out_path_qt}"
)

print("QuickTime-compatible file:", out_path_qt)
