# ==========================================================
# Person1 Tracking + Pose Estimation + CSV/NPZ Export
# ==========================================================
# This script performs the full person-selection and pose-estimation pipeline.
# 1. Loads bed metadata (best frame, bounding box, gate polygon)
# 2. Runs YOLO pose estimation on every frame
# 3. Selects the main patient (“person1”) inside the bed gate
# 4. Tracks person1, handling missing frames and recovery
# 5. Draws skeleton overlays and saves the output video
# 6. Exports all keypoints to CSV and NPZ for later analysis
# 7. Updates the metadata JSON with tracking results
# 
# Notes:
# - Only the human pose is estimated (YOLO pose model). No background objects
#   (bed, staff, equipment) are estimated in this script.
# - The “gate” (polygon or rectangle) filters which detected person belongs to
#   the bed region, improving robustness in multi-person scenes.
# - Missing segments are logged and summarized.
# ==========================================================

import os, csv, json, math
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
from ultralytics import YOLO

# ========= I/O paths =========
# Input video, metadata, and output file locations.
# - out_try_path / out_qt_path: skeleton-overlay videos (raw + QuickTime-safe)
# - csv_path: keypoint time-series (17 joints × x,y,confidence)
# - log_path: missing/recovery events for person1
# - npz_out: cleaned keypoints for downstream signal analysis
video_path   = '/content/****.mp4'
json_path    = '/content/bed_best.json'     
meta_json_path = '/content/meta.json'   # すべての統合情報を格納するメタファイル
out_try_path = '/content/person1_lock_track.mp4'
out_qt_path  = '/content/person1_lock_track_qt.mp4'
csv_path     = '/content/person1_kpts_locked.csv'
log_path     = '/content/person1_events.log'
npz_out      = '/content/person1_clean_kpts.npz'

# ========= Hyperparameters =========
# GATE_EXPAND_PX: extra padding (px) added when a rectangular gate is used
# MISS_MAX: max consecutive frames where person1 may disappear before switching state
#           (e.g., 60 frames ≈ 2 seconds at 30 fps)
# CONF_MIN: minimum keypoint confidence; low-confidence points are marked as NaN
GATE_EXPAND_PX = 40
MISS_MAX       = 60      
CONF_MIN       = 0.15

# ========= 事前情報の読み込み（新旧両フォーマット対応） =========
#ベッド検出のメタ情報とゲート情報（gate_config）」を読み込んで、人物選定（person1）に使うための基礎データを復元するための関数 
#best_frame_idx: ベッドが最もきれいに写ったフレーム番号
#bed_xyxy: ベッド検出ボックス（x1, y1, x2, y2）
#bed_cx, bed_cy: ベッド中心（人物選定で距離計算に使う）
#gate_poly: そのフレームで作ったゲート領域（人選びに使う。None の場合は矩形に切り替える）


def load_meta(json_path):
    if not os.path.exists(json_path):
        raise RuntimeError("Meta JSON not found: " + json_path)
    with open(json_path, "r") as f:
        meta = json.load(f)

    best_frame = int(meta["best_frame"])
    bed_xyxy   = np.array(meta["bed_xyxy"], dtype=float)
    bed_cxcy   = tuple(meta["bed_cxcy"])
    gate_poly  = None

    if "gate_polygon" in meta and meta["gate_polygon"]:
        gate_poly = np.array(meta["gate_polygon"], dtype=np.float32).reshape(-1,1,2)

    return best_frame, bed_xyxy, bed_cxcy, gate_poly

best_frame_idx, bed_xyxy, (bed_cx, bed_cy), gate_poly = load_meta(json_path)
gate_mode = "poly" if gate_poly is not None else "rect"

# ========= モデル =========
pose_model = YOLO('yolo11x-pose.pt')

# ========= ユーティリティ =========
def center_from_kpts(kp_xy):
    valid = ~np.isnan(kp_xy).any(axis=1)
    if not np.any(valid): return None
    return np.nanmean(kp_xy[valid], axis=0)

def in_gate_poly(cx, cy, gate_polygon):
    return cv2.pointPolygonTest(gate_polygon, (float(cx), float(cy)), False) >= 0

# ========= スケルトン描画 =========
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
            cv2.line(img, (int(xa),int(ya)), (int(xb),int(yb)), color, thickness, cv2.LINE_AA)

# ========= 動画IO =========
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f'VideoCaptureが開けません: {video_path}')
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if gate_mode == "rect":
    x1,y1,x2,y2 = bed_xyxy.copy()
    x1 -= GATE_EXPAND_PX; y1 -= GATE_EXPAND_PX
    x2 += GATE_EXPAND_PX; y2 += GATE_EXPAND_PX
    x1 = max(0, min(W-1, int(round(x1))))
    y1 = max(0, min(H-1, int(round(y1))))
    x2 = max(0, min(W-1, int(round(x2))))
    y2 = max(0, min(H-1, int(round(y2))))
    gate_poly = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32).reshape(-1,1,2)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_try_path, fourcc, fps, (W, H))
if not out.isOpened():
    raise RuntimeError('VideoWriterが開けません')

# ========= person1 状態 =========
state = "INIT"; miss_count = 0
person1_idx = None
missing_segments = []
missing_start = None

# ========= CSV準備 =========
kpt_headers = [f"k{i}_{axis}" for i in range(17) for axis in ('x','y','c')]
with open(csv_path, 'w', newline='') as fcsv, open(log_path, 'w') as flog:
    writer = csv.writer(fcsv)
    writer.writerow(['frame', 'state', 'person1_idx'] + kpt_headers)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        res = pose_model.predict(frame, verbose=False, conf=CONF_MIN)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if (res.boxes is not None and len(res.boxes)>0) else np.zeros((0,4))
        kpts  = res.keypoints.data.cpu().numpy() if (res.keypoints is not None and res.keypoints.data is not None) else np.zeros((0,17,3))
        N = boxes.shape[0]

        candidates = []
        for i in range(N):
            kp_xy = kpts[i,:,:2] if i < kpts.shape[0] else np.full((17,2), np.nan)
            kp_c  = kpts[i,:,2]   if i < kpts.shape[0] else np.full((17,), np.nan)
            low = kp_c < CONF_MIN; kp_xy = kp_xy.copy(); kp_xy[low] = np.nan
            center = center_from_kpts(kp_xy)
            if center is None:
                x1,y1,x2,y2 = boxes[i]; center = np.array([(x1+x2)/2.0, (y1+y2)/2.0])
            if in_gate_poly(center[0], center[1], gate_poly):
                candidates.append({'i': i, 'center': center, 'kp_xy': kp_xy, 'kp_c': kp_c})

        chosen = None
        if len(candidates)>0:
            d_bed = [np.hypot(c['center'][0]-bed_cx, c['center'][1]-bed_cy) for c in candidates]
            j = int(np.argmin(d_bed)); chosen = candidates[j]

        if chosen is not None:
            person1_idx = chosen['i']
            state = "LOCKED"
            miss_count = 0
            if missing_start is not None:
                missing_segments.append((missing_start, frame_idx-1))
                dur = (frame_idx - missing_start)/fps
                print(f"[RECOVER] frame={frame_idx} (欠落 {dur:.2f}s)")
                missing_start = None
        else:
            miss_count += 1
            if miss_count == 1:
                missing_start = frame_idx
                print(f"[MISS] frame={frame_idx}")
            if miss_count > MISS_MAX:
                state = "MISSING"

        vis = frame.copy()
        cv2.polylines(vis, [gate_poly.astype(np.int32)], True, (0,180,0), 2)
        row_k = [np.nan]*(17*3)
        if person1_idx is not None and N > person1_idx:
            kp_xy = kpts[person1_idx,:,:2].copy()
            kp_c = kpts[person1_idx,:,2]
            low = kp_c < CONF_MIN; kp_xy[low] = np.nan
            draw_skeleton(vis, kp_xy, color=(0,0,255), thickness=2)
            flat = [float(x) if np.isfinite(x) else np.nan for xy in kp_xy for x in xy]
            flat += [float(c) if np.isfinite(c) else np.nan for c in kp_c]
            row_k = flat

        out.write(vis)
        writer.writerow([frame_idx, state, person1_idx] + row_k)
        frame_idx += 1

cap.release(); out.release()

# 欠落終了処理
if missing_start is not None:
    missing_segments.append((missing_start, frame_idx-1))

if missing_segments:
    print("\n=== 欠落サマリ ===")
    total_missing = sum((e - s + 1) / fps for s,e in missing_segments)
    for s,e in missing_segments:
        dur = (e - s + 1) / fps
        print(f"欠落: frame {s}–{e} ({dur:.2f}s)")
    print(f"総欠落時間: {total_missing:.2f}s / {frame_idx/fps:.2f}s "
          f"({100*total_missing/(frame_idx/fps):.1f}%)")
else:
    print("\n=== 欠落なし ===")

# ========= QuickTime互換 =========
os.system(f"ffmpeg -y -i {out_try_path} -vcodec libx264 -pix_fmt yuv420p "
          f"-profile:v baseline -level 3.0 -movflags +faststart {out_qt_path}")

print("[OK] Video:", out_qt_path)
print("[OK] CSV  :", csv_path)
print("[OK] Log  :", log_path)

# ========= CSV → NPZ 変換 =========
df = pd.read_csv(csv_path)
N = len(df)
time_all = np.arange(N, dtype=float) / float(fps)
def xy_of(k):
    x = df[f'k{k}_x'].to_numpy(float)
    y = df[f'k{k}_y'].to_numpy(float)
    return np.stack([x, y], axis=1)
nose = xy_of(0); LS=xy_of(5); RS=xy_of(6); LH=xy_of(11); RH=xy_of(12)
np.savez(npz_out, time_all=time_all, fps=float(fps), LS=LS, RS=RS, LH=LH, RH=RH, nose=nose)
print("[OK] NPZ :", npz_out)

# ========= Load existing meta if available =========
meta = {}
if os.path.exists(meta_json_path):
    try:
        with open(meta_json_path, 'r') as f:
            meta = json.load(f)
    except:
        print("[WARN] Failed to load existing meta.json")

meta['best_frame'] = int(best_frame_idx)
meta['bed_xyxy']   = [float(x) for x in bed_xyxy]
meta['bed_cxcy']   = [float(bed_cx), float(bed_cy)]
if gate_poly is not None:
    meta['gate_polygon'] = gate_poly.reshape(-1,2).astype(float).tolist()
if 'margins' in gate_cfg:
    meta['margins'] = gate_cfg['margins']
meta['person1_idx'] = int(person1_idx) if person1_idx is not None else None
meta['missing_segments'] = [[int(s), int(e)] for s,e in missing_segments]
total_missing_sec = sum((e - s + 1) / fps for s,e in missing_segments)
meta['missing_total_sec'] = float(total_missing_sec)
meta['video_path'] = str(video_path)
meta['fps'] = float(fps)
meta['outputs'] = {
    'overlay_video': str(out_qt_path),
    'csv': str(csv_path),
    'npz': str(npz_out),
    'log': str(log_path)
}

with open(meta_json_path, 'w') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"[OK] Updated meta JSON: {meta_json_path}")
