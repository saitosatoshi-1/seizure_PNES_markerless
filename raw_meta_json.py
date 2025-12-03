# ==========================================================
# Person1 Tracking + Pose Estimation + NPZ Export
# ==========================================================
# This script performs the full person-selection and pose-estimation pipeline.
# 1. Loads bed metadata (best frame, bounding box, gate polygon)
# 2. Runs YOLO pose estimation for each frame
# 3. Selects the main patient ("person1") inside the bed gate
# 4. Tracks person1, handling missing frames and recovery
# 5. Overlays skeletons and writes an output video
# 6. Saves all keypoints directly to NPZ (no CSV)
# 7. Updates meta JSON with tracking results
#
# Notes:
# - Only human pose is estimated (YOLO pose model).
# - No background objects (bed/staff/equipment) are estimated here.
# - The gate polygon filters which detection belongs to the bed region.
# - Missing segments (lost tracking) are logged and summarized.
# ==========================================================

import os, json
import numpy as np
import cv2
from ultralytics import YOLO

# ========= I/O Paths =========
video_path      = "/content/****.mp4"
json_path       = "/content/bed_best.json"     # input metadata
meta_json_path  = "/content/meta.json"         # output combined metadata
out_try_path    = "/content/person1_lock_track.mp4"
out_qt_path     = "/content/person1_lock_track_qt.mp4"
log_path        = "/content/person1_events.log"
npz_out         = "/content/person1_clean_kpts.npz"

# ========= Hyperparameters =========
GATE_EXPAND_PX = 40
MISS_MAX       = 60
CONF_MIN       = 0.15

# ========= Load Metadata (bed_best.json) =========
def load_meta(json_path):
    if not os.path.exists(json_path):
        raise RuntimeError("Meta JSON not found: " + json_path)
    with open(json_path, "r") as f:
        meta = json.load(f)

    best_frame = int(meta["best_frame"])
    bed_xyxy   = np.array(meta["bed_xyxy"], dtype=float)
    bed_cxcy   = tuple(meta["bed_cxcy"])

    gate_poly = None
    if "gate_polygon" in meta and meta["gate_polygon"]:
        gate_poly = np.array(meta["gate_polygon"], dtype=np.float32).reshape(-1, 1, 2)

    return best_frame, bed_xyxy, bed_cxcy, gate_poly

best_frame_idx, bed_xyxy, (bed_cx, bed_cy), gate_poly = load_meta(json_path)
gate_mode = "poly" if gate_poly is not None else "rect"

# ========= YOLO Pose Model =========
pose_model = YOLO("yolo11x-pose.pt")

# ========= Utilities =========
def center_from_kpts(kp_xy):
    valid = ~np.isnan(kp_xy).any(axis=1)
    if not np.any(valid):
        return None
    return np.nanmean(kp_xy[valid], axis=0)

def in_gate_poly(cx, cy, gate_polygon):
    return cv2.pointPolygonTest(gate_polygon, (float(cx), float(cy)), False) >= 0

# ========= Skeleton Drawing =========
SKELETON = [
    (0,1),(1,3),(0,2),(2,4),
    (5,7),(7,9),(6,8),(8,10),
    (5,6),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16)
]

def draw_skeleton(img, kp_xy, color=(0,0,255), thickness=2):
    for a, b in SKELETON:
        xa, ya = kp_xy[a]; xb, yb = kp_xy[b]
        if not (np.isnan([xa,ya]).any() or np.isnan([xb,yb]).any()):
            cv2.line(img, (int(xa),int(ya)), (int(xb),int(yb)),
                     color, thickness, cv2.LINE_AA)

# ========= Open Video =========
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Failed to open video: " + video_path)

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Rectangle fallback if polygon not available
if gate_mode == "rect":
    x1,y1,x2,y2 = bed_xyxy.copy()
    x1 -= GATE_EXPAND_PX; y1 -= GATE_EXPAND_PX
    x2 += GATE_EXPAND_PX; y2 += GATE_EXPAND_PX
    x1 = max(0, min(W-1, int(round(x1))))
    y1 = max(0, min(H-1, int(round(y1))))
    x2 = max(0, min(W-1, int(round(x2))))
    y2 = max(0, min(H-1, int(round(y2))))
    gate_poly = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],
                         dtype=np.float32).reshape(-1,1,2)

# ========= Output Video Writer =========
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(out_try_path, fourcc, fps, (W, H))
if not out.isOpened():
    raise RuntimeError("Failed to open VideoWriter")

# ========= Tracking State =========
state = "INIT"
miss_count = 0
person1_idx = None
missing_segments = []
missing_start = None

# ========= NPZ Buffers =========
all_kpts = []     # (frame, 17, 3)
all_state = []    # "INIT", "LOCKED", "MISSING"
all_idx   = []    # person1_idx per frame

# ========= Log File =========
flog = open(log_path, "w")

# ========= Frame Loop =========
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    res = pose_model.predict(frame, verbose=False, conf=CONF_MIN)[0]
    boxes = res.boxes.xyxy.cpu().numpy() if (res.boxes is not None and len(res.boxes)>0) else np.zeros((0,4))
    kpts  = res.keypoints.data.cpu().numpy() if (res.keypoints is not None and res.keypoints.data is not None) else np.zeros((0,17,3))
    N = boxes.shape[0]

    # ---- Select candidates within gate ----
    candidates = []
    for i in range(N):
        kp_xy = kpts[i,:,:2] if i < kpts.shape[0] else np.full((17,2), np.nan)
        kp_c  = kpts[i,:,2]  if i < kpts.shape[0] else np.full((17,), np.nan)

        kp_xy = kp_xy.copy()
        kp_xy[kp_c < CONF_MIN] = np.nan

        center = center_from_kpts(kp_xy)
        if center is None:
            x1,y1,x2,y2 = boxes[i]
            center = np.array([(x1+x2)/2, (y1+y2)/2])

        if in_gate_poly(center[0], center[1], gate_poly):
            candidates.append({
                "i": i, "center": center, "kp_xy": kp_xy, "kp_c": kp_c
            })

    # ---- Choose closest to bed center ----
    chosen = None
    if len(candidates) > 0:
        d = [np.hypot(c["center"][0]-bed_cx, c["center"][1]-bed_cy) for c in candidates]
        chosen = candidates[int(np.argmin(d))]

    # ---- Tracking state machine ----
    if chosen is not None:
        person1_idx = chosen["i"]
        state = "LOCKED"
        miss_count = 0

        if missing_start is not None:
            missing_segments.append((missing_start, frame_idx-1))
            dur = (frame_idx - missing_start) / fps
            flog.write(f"[RECOVER] frame={frame_idx} ({dur:.2f}s missing)\n")
            missing_start = None

    else:
        miss_count += 1
        if miss_count == 1:
            missing_start = frame_idx
            flog.write(f"[MISS] frame={frame_idx}\n")
        if miss_count > MISS_MAX:
            state = "MISSING"

    # ---- Draw skeleton ----
    vis = frame.copy()
    cv2.polylines(vis, [gate_poly.astype(np.int32)], True, (0,180,0), 2)

    # Default (no person1)
    row_k = np.full((17,3), np.nan, dtype=float)

    if person1_idx is not None and N > person1_idx:
        kp_xy = kpts[person1_idx,:,:2].copy()
        kp_c  = kpts[person1_idx,:,2]
        kp_xy[kp_c < CONF_MIN] = np.nan
        draw_skeleton(vis, kp_xy)
        row_k = np.stack([kp_xy[:,0], kp_xy[:,1], kp_c], axis=1)

    # ---- Save video + NPZ buffers ----
    out.write(vis)
    all_kpts.append(row_k)
    all_state.append(state)
    all_idx.append(person1_idx)

    frame_idx += 1

# ========= Clean Up Video =========
cap.release()
out.release()
flog.close()

# ========= Missing summary =========
if missing_start is not None:
    missing_segments.append((missing_start, frame_idx-1))

# ========= Save NPZ =========
all_kpts = np.array(all_kpts, dtype=float)   # (F,17,3)
time_all = np.arange(len(all_kpts)) / fps

np.savez(
    npz_out,
    kpts=all_kpts,
    time_all=time_all,
    fps=float(fps),
    state=np.array(all_state),
    person1_idx=np.array(all_idx),
)
print("[OK] NPZ saved:", npz_out)

# ========= QuickTime Re-encode =========
os.system(
    f"ffmpeg -y -i {out_try_path} -vcodec libx264 -pix_fmt yuv420p "
    f"-profile:v baseline -level 3.0 -movflags +faststart {out_qt_path}"
)
print("[OK] Video:", out_qt_path)

# ========= Update meta.json =========
meta = {}
if os.path.exists(meta_json_path):
    try:
        with open(meta_json_path, "r") as f:
            meta = json.load(f)
    except:
        print("[WARN] Failed to load existing meta.json")

meta["best_frame"] = int(best_frame_idx)
meta["bed_xyxy"]   = bed_xyxy.astype(float).tolist()
meta["bed_cxcy"]   = [float(bed_cx), float(bed_cy)]
meta["gate_polygon"] = gate_poly.reshape(-1,2).astype(float).tolist()
meta["person1_idx"]  = int(person1_idx) if person1_idx is not None else None
meta["missing_segments"] = [[int(s), int(e)] for s,e in missing_segments]
meta["missing_total_sec"] = float(sum((e - s + 1)/fps for s,e in missing_segments))
meta["video_path"] = str(video_path)
meta["fps"]        = float(fps)
meta["outputs"] = {
    "overlay_video": out_qt_path,
    "npz": npz_out,
    "log": log_path,
}

with open(meta_json_path, "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print("[OK] Updated meta JSON:", meta_json_path)
