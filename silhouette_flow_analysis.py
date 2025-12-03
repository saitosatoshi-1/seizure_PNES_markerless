# ==========================================================
# Skeleton + Silhouette Optical Flow (person1, optimized)
# ----------------------------------------------------------
# This script combines:
#   1) Cleaned skeleton keypoints (person1_clean_kpts.npz)
#   2) Human segmentation (YOLO11-seg)
#   3) Bed / gate zones (gate_config.json)
#
# Main features
#   - gate_config.json provides priority / allow / exclude zones
#     + “zone hysteresis” to keep the mask inside the bed region.
#   - Skeleton-based torso ROI (“capsule” from shoulders to hips)
#     is used to filter segmentation candidates → reduces jumps to
#     objects / other people.
#   - At the beginning, the tracker is strict: person1 must be
#     close to the anchor (hip midpoint) inside priority + ROI.
#   - Time alignment: video time (CAP_PROP_POS_MSEC) is mapped
#     to skeleton time by np.searchsorted (+ optional time shift).
#   - Upper vs lower body: the full mask is split by a line
#     orthogonal to the hip axis (LH–RH).
#
# Outputs
#   - Video with overlay (segmentation, zones, upper/lower body,
#     heatmap of flow magnitude)
#   - CSV: frame-wise flow metrics (raw + cleaned)
#   - Stats CSV: cleaning statistics for each flow series
#
# Typical upstream dependency
#   - person1_clean_kpts.npz: produced by a “person1 pose cleaning”
#     script (jump removal, interpolation, smoothing, etc.)
#   - gate_config.json: produced by a separate bed / gate builder.
# ==========================================================

import os, json
import numpy as np
import cv2
from ultralytics import YOLO
import pandas as pd

# =========================
# I/O paths
# =========================
video_path   = '/content/FBTCS_silhouette_qt.mp4'     # input silhouette video (QuickTime-safe)
npz_path     = '/content/person1_clean_kpts.npz'      # cleaned skeleton NPZ (LS, RS, LH, RH, etc.)
gate_json    = '/content/gate_config.json'            # gate + zone configuration
csv_path     = '/content/person1_silhouette_flow.csv' # output per-frame flow metrics
stats_csv    = '/content/person1_flow_stats.csv'      # output cleaning stats per series
out_try_path = '/content/person1_silhouette_seg_clean.mp4'     # raw overlay video
out_qt_path  = '/content/person1_silhouette_seg_clean_qt.mp4'  # QuickTime-safe overlay

# =========================
# Visualization colors
# =========================
COLOR_UPPER_FILL = (60, 200, 60)    # upper-body fill
COLOR_LOWER_FILL = (200, 60, 200)   # lower-body fill
COLOR_SPLIT      = (0, 255, 255)    # hip-based split line
COLOR_SEL        = (80, 255, 80)    # selected contour outline
COLOR_CNT        = (170, 170, 170)  # other contours
COLOR_ZONE_PRIO  = (0, 200, 255)    # priority zone outline
COLOR_ZONE_ALLOW = (120, 120, 255)  # allow zone outline
COLOR_ZONE_EXCL  = (60, 60, 60)     # exclude zone outline

ALPHA_UP   = 0.40   # alpha blending for upper mask
ALPHA_LOW  = 0.40   # alpha blending for lower mask
ALPHA_HEAT = 0.30   # alpha for flow heatmap overlay
HEAT_PCTL  = 95     # percentile for heatmap scaling

DRAW_ALL_CONTOURS = True
SHOW_HEAT         = True
SHOW_ZONES        = True

# =========================
# Hyperparameters
# =========================
CONF_MIN        = 0.25
KERNEL_MORPH    = (3, 3)
T0_SEC          = 0.0        # ignore frames before this time in stats/cleaning
MISS_MAX_SEC    = 1.0        # max continuous miss for cleaning window
MEDIAN_WIN_FR   = 2          # median window for post-smoothing
PCLIP_LOW       = 1          # percentile low for clipping outliers
PCLIP_HIGH      = 99         # percentile high
USE_POS_MSEC    = True       # use CAP_PROP_POS_MSEC for time
ANCHOR_SHIFT_SEC = 0.0       # manual fine time shift between video and skeleton

# Anchor strict warm-up period (early frames)
LOCK_WARMUP_SEC      = 3.0
ANCHOR_RADIUS_RATIO  = 0.10  # how close to hip-anchor (relative to image diagonal)
ANCHOR_CENT_MAX_RATIO = 0.12

# Stickiness & shape constraints for segmentation selection
IOU_KEEP         = 0.35    #前フレームとIoU（重なり）が35%以上 → 同じ人物とみなして保持
DIST_KEEP_RATIO  = 0.20    #中心の移動距離が画面対角の20%以内 → 同じ人物として継続
IOU_SWITCH       = 0.15    #距離が大きい → 別の物体に切り替えるべき
DIST_SWITCH_RATIO = 0.35   #中心距離が35%を超える → 大きく移動しすぎ → 選択を切り替え
STICKY_FRAMES    = 8       #最大8フレームは「同じ人物」と粘る。
ZONE_STICKY_FRAMES = 12    #ゾーン（優先領域）を８フレーム以上維持しやすくする。

MIN_AREA_RATIO   = 0.005   # min mask area relative to frame
MIN_HEIGHT_RATIO = 0.15    # min mask height relative to frame
MAX_ASPECT       = 6.0     # max aspect ratio width/height or height/width

# Skeleton-based torso ROI capsule
POSE_OVERLAP_MIN    = 0.55   # min overlap ratio between candidate mask and capsule
POSE_BONUS          = 2.0    # score bonus when mask aligns with capsule
POSE_STRICT_WARMUP  = True   # during warm-up, reject masks with poor capsule overlap
CAPSULE_RADIUS_SCALE = 0.6   # torso capsule radius = median shoulder/hip width * scale

# =========================
# Segmentation model
# =========================
seg_model = YOLO('yolo11s-seg.pt')

# =========================
# Video I/O
# =========================
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f'Failed to open video: {video_path}')

fps = cap.get(cv2.CAP_PROP_FPS)
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    out_try_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (W, H)
)

YY, XX = np.mgrid[0:H, 0:W]
k_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, KERNEL_MORPH)

# =========================
# Load cleaned skeleton NPZ
# =========================
dat      = np.load(npz_path)
time_all = dat["time_all"]
LS, RS   = dat["LS"], dat["RS"]
LH, RH   = dat["LH"], dat["RH"]

# =========================
# Check available resolution keys
# =========================
print("[NPZ] keys:", dat.files)

has_frame_w = "frame_w" in dat.files
has_ref_w   = "ref_w"   in dat.files
has_frame_h = "frame_h" in dat.files
has_ref_h   = "ref_h"   in dat.files

print(f"[NPZ] frame_w: {has_frame_w}, ref_w: {has_ref_w}")
print(f"[NPZ] frame_h: {has_frame_h}, ref_h: {has_ref_h}")

# =========================
# Determine reference resolution
# =========================
if has_frame_w and has_frame_h:
    refW = dat["frame_w"].item()
    refH = dat["frame_h"].item()
    print(f"[NPZ] Using frame_w/frame_h: ({refW}, {refH})")

elif has_ref_w and has_ref_h:
    refW = dat["ref_w"].item()
    refH = dat["ref_h"].item()
    print(f"[NPZ] Using ref_w/ref_h: ({refW}, {refH})")

else:
    # fallback: assume NPZ was generated with current video size
    refW, refH = W, H
    print(f"[NPZ] No resolution info found — using current video size ({W}, {H})")

# =========================
# Rescale skeleton coordinates if needed
# =========================
if (refW, refH) != (W, H):
    print(f"[NPZ] Rescaling skeleton: ({refW}, {refH}) → ({W}, {H})")
    sx, sy = W / refW, H / refH
    for A in (LS, RS, LH, RH):
        A[:, 0] *= sx
        A[:, 1] *= sy
else:
    print("[NPZ] No rescaling needed.")

hip_mid      = (LH + RH) / 2.0
shoulder_mid = (LS + RS) / 2.0
shoulder_w_med = float(np.nanmedian(np.linalg.norm(RS - LS, axis=1)))
hip_w_med      = float(np.nanmedian(np.linalg.norm(RH - LH, axis=1)))

# =========================
# Load gate_config.json (zones)
#“gate_config.json に描かれている領域（ポリゴン）を、動画サイズのマスク画像に変換している”
#priority（優先） → 患者である可能性が最も高い領域
#allow（許可） → 患者が存在してよい領域
#exclude（禁止） → 患者が絶対に存在しない領域（スタッフや機器）
#この3つの領域を「画像と同じ大きさの2Dマップ」に変換する処理です。
# =========================
def _as_polys(obj):
    """
    Convert various JSON formats into a list of polygons.
    Allowed patterns:
      - [[x, y], ...]
      - [[[x, y], ...], ...] (multiple polygons)
      - or wrapped in keys such as "priority", "polygon", etc.
    """
    if obj is None:
        return []
    if isinstance(obj, dict):
        for k in ("priority", "allow", "exclude", "bed_poly", "polygon", "gate", "poly", "points"):
            if k in obj and isinstance(obj[k], (list, tuple)):
                obj = obj[k]
                break
    if isinstance(obj, list) and obj and isinstance(obj[0], (list, tuple)):
        if obj and isinstance(obj[0][0], (list, tuple)):
            return [np.array(p, dtype=float) for p in obj]  # multiple polygons
        return [np.array(obj, dtype=float)]
    return []

def load_gate_masks(json_path, W, H, refW=None, refH=None):
    """
    Load priority / allow / exclude masks (boolean arrays).
    Polygons are rescaled if the JSON was generated at a different resolution.
    """
    if not os.path.isfile(json_path):
        return None, None, None

    with open(json_path, 'r') as f:
        js = json.load(f)

    # Reference resolution in JSON (if stored)
    jrefW = js.get("ref_w", js.get("frame_w", refW))
    jrefH = js.get("ref_h", js.get("frame_h", refH))
    if jrefW and jrefH:
        refW, refH = jrefW, jrefH

    def scale_poly(P):
        if P is None:
            return None
        Q = P.copy()
        if (refW and refH) and ((refW, refH) != (W, H)):
            Q[:, 0] *= W / float(refW)
            Q[:, 1] *= H / float(refH)
        return Q

    pri_list   = _as_polys(js.get("priority")) or _as_polys(js.get("bed_poly")) or _as_polys(js.get("polygon"))
    allow_list = _as_polys(js.get("allow"))
    excl_list  = _as_polys(js.get("exclude"))

    # allow: if not specified, default to full frame
    if not allow_list:
        allow_mask = np.ones((H, W), np.uint8)
    else:
        allow_mask = np.zeros((H, W), np.uint8)
        for P in allow_list:
            P = scale_poly(P)
            cv2.fillPoly(allow_mask, [P.astype(np.int32)], 1)

    pri_mask = np.zeros((H, W), np.uint8)
    for P in pri_list or []:
        P = scale_poly(P)
        cv2.fillPoly(pri_mask, [P.astype(np.int32)], 1)

    exc_mask = np.zeros((H, W), np.uint8)
    for P in excl_list or []:
        P = scale_poly(P)
        #OpenCVでポリゴン内部を1（True）で塗りつぶしてH×Wのマスク画像を作る。
        cv2.fillPoly(exc_mask, [P.astype(np.int32)], 1)

    return pri_mask.astype(bool), allow_mask.astype(bool), exc_mask.astype(bool)

priority_mask, allow_mask, exclude_mask = load_gate_masks(gate_json, W, H, refW, refH)

# =========================
# Utility functions
#人物シルエットの重心を計算
#輪郭を描画
#色を重ねて可視化
#ヒートマップを重ねる
#マスク同士の IoU（似ている度）を計算
#allow / priority / exclude のゾーン判定
# =========================

#マスク領域の重心を返す
def mcent(m):
    """Return centroid (x, y) of a boolean mask, or None if empty."""
    #マスクmの中でTrueになっているピクセルを取り出す
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())

#マスクの輪郭を画像に描画
def draw_cnt(img, m, col, th=2):
    """Draw contour(s) of a boolean mask."""
    if m is None or not m.any():
        return
    m8 = (m.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(img, cnts, -1, col, th)

#priority/allow/excludeの枠を描画
def draw_poly_mask(img, mask, color, thickness=2):
    """Draw polygon outlines from a boolean mask (for zone visualization)."""
    if mask is None:
        return
    m8 = (mask.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(img, cnts, -1, color, thickness)

def overlay(img, m, col, a):
    """Blend color `col` onto image where mask m is True."""
    if m is None or not m.any():
        return
    ov = img.copy()
    ov[m] = (ov[m] * (1 - a) + np.array(col) * a).astype(np.uint8)
    img[:] = ov

def heat_overlay(img, mag, m, a, pctl):
    """Flow magnitude heatmap overlay inside mask m."""
    if m is None or not m.any():
        return
    if np.isfinite(mag[m]).any():
        vmax = np.nanpercentile(mag[m], pctl)
    else:
        vmax = 1.0
    vmax = max(vmax, 1e-6)
    heat = (np.clip(mag / vmax, 0, 1) * 255).astype(np.uint8)
    cmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    ov = img.copy()
    ov[m] = (cmap[m] * a + img[m] * (1 - a)).astype(np.uint8)
    img[:] = ov

#前のフレームのmaskと今の候補maskが似ているか判定
def iou(a, b):
    """IoU of two boolean masks."""
    if a is None or b is None:
        return 0.0
    inter = np.logical_and(a, b).sum()
    uni   = np.logical_or(a, b).sum()
    return inter / (uni + 1e-6)

def clip_to_allow(m):
    """Force mask to stay inside the allow_mask (if defined)."""
    return np.logical_and(m, allow_mask) if allow_mask is not None else m

#マスクの形が逸脱していないか?
def is_ok_shape(m):
    """Basic sanity check on mask shape (area/height/aspect ratio)."""
    area = int(m.sum())
    if area < MIN_AREA_RATIO * (W * H):
        return False
    ys, xs = np.nonzero(m)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    w = max(1, x2 - x1 + 1)
    h = max(1, y2 - y1 + 1)
    if max(w / h, h / w) > MAX_ASPECT:
        return False
    if h < MIN_HEIGHT_RATIO * H:
        return False
    return True

#priority/allow/noneを判定
def zone_of_mask(m):
    """Return which zone the mask overlaps: priority / allow / none."""
    if m is None:
        return "none"
    if priority_mask is not None and (m & priority_mask).any():
        return "priority"
    if allow_mask is not None and (m & allow_mask).any():
        return "allow"
    return "none"

def candidate_zone_bonus(m):
    """Return a score bonus based on zones; None if excluded."""
    if exclude_mask is not None and (m & exclude_mask).any():
        return None
    if priority_mask is not None and (m & priority_mask).any():
        return 1.0
    if allow_mask is not None and (m & allow_mask).any():
        return 0.2
    return None

# =========================
# Skeleton torso capsule ROI
#肩と股関節の位置から胴体カプセル領域を作る
# =========================
def capsule_mask_from_pose(
    W, H,
    shoulder_xy,
    hip_xy,
    shoulder_w_med,
    hip_w_med,
    scale_rad=CAPSULE_RADIUS_SCALE,
):
    """
    Build a torso capsule ROI connecting shoulder and hip centers.
    Approximation:
      - circle around shoulders (radius ~ shoulder width)
      - circle around hips (radius ~ hip width)
      - thick line between them
    """
    canvas = np.zeros((H, W), np.uint8)
    p1 = np.asarray(shoulder_xy, float)
    p2 = np.asarray(hip_xy, float)
    if not (np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))):
        return canvas.astype(bool)

    r1 = max(6.0, float(shoulder_w_med) * scale_rad)
    r2 = max(6.0, float(hip_w_med) * scale_rad)

    cv2.circle(canvas, (int(round(p1[0])), int(round(p1[1]))), int(round(r1)), 255, -1)
    cv2.circle(canvas, (int(round(p2[0])), int(round(p2[1]))), int(round(r2)), 255, -1)
    thickness = max(1, int(round(r1 + r2)))
    cv2.line(
        canvas,
        (int(round(p1[0])), int(round(p1[1]))),
        (int(round(p2[0])), int(round(p2[1]))),
        255,
        thickness,
    )
    M = (canvas > 0)
    return np.logical_and(M, allow_mask) if allow_mask is not None else M

#この関数は「候補マスクが患者の胴体 ROI のどれだけを占めているか」を返す。
#値が大きいほど、その候補は本物の患者である可能性が高い。
def overlap_ratio(mask_roi, mask_cand):
    """Overlap ratio = intersection / candidate area."""
    if mask_roi is None or mask_cand is None:
        return 0.0
    inter = np.logical_and(mask_roi, mask_cand).sum()
    denom = mask_cand.sum() + 1e-6
    return float(inter) / float(denom)


# =========================
# Candidate selection (IoU + distance + anchor + ROI + zone hysteresis)
# =========================
def select_mask(prev_m, prev_zone, cand_list, t_sec, anchor_xy, sticky, zone_sticky, pose_roi):
    """
    Select the best segmentation mask among candidates by:
      - IoU and distance from previous mask
      - distance from anchor (hip_mid)
      - overlap with skeleton torso ROI
      - priority / allow zones with hysteresis
    """
    diag = np.hypot(W, H)
    warmup = (t_sec <= LOCK_WARMUP_SEC) and (anchor_xy is not None)
    best = None
    best_score = -1.0
    prev_c = mcent(prev_m) if prev_m is not None else None
    r = ANCHOR_RADIUS_RATIO * diag if warmup else None

    for m in cand_list:
        m = clip_to_allow(m)
        if exclude_mask is not None and (m & exclude_mask).any():
            continue
        if not is_ok_shape(m):
            continue

        # 1) Skeleton torso ROI overlap
        pose_ov = overlap_ratio(pose_roi, m) if pose_roi is not None else 0.0
        if pose_roi is not None:
            # strict filter during warm-up or strict mode
            if pose_ov < POSE_OVERLAP_MIN and (POSE_STRICT_WARMUP or warmup):
                continue

        c = mcent(m)
        if c is None:
            continue

        # 2) Warm-up: must be inside priority + close to anchor
        if warmup:
            if priority_mask is not None and not (m & priority_mask).any():
                continue
            if np.linalg.norm(np.array(c) - np.array(anchor_xy)) > r:
                continue
            if (np.linalg.norm(np.array(c) - np.array(anchor_xy)) / (diag + 1e-6)) > ANCHOR_CENT_MAX_RATIO:
                continue

        # 3) Base score from IoU + distance to previous
        j = iou(prev_m, m) if prev_m is not None else 0.0
        d_ratio = (
            1.0
            if prev_c is None
            else np.linalg.norm(np.array(c) - np.array(prev_c)) / (diag + 1e-6)
        )
        score = 2.0 * j + (1.0 - min(1.0, d_ratio))

        # 4) Anchor proximity
        if anchor_xy is not None:
            da = np.linalg.norm(np.array(c) - np.array(anchor_xy)) / (diag + 1e-6)
            score += 0.8 * (1.0 - min(1.0, da))

        # 5) Zone bonus (priority / allow)
        z_bonus = candidate_zone_bonus(m)
        if z_bonus is None:
            continue
        score += z_bonus

        # 6) ROI overlap bonus / penalty
        if pose_roi is not None:
            score += POSE_BONUS * pose_ov
            if pose_ov < POSE_OVERLAP_MIN:
                score -= 0.5

        # 7) Zone hysteresis
        cand_zone = zone_of_mask(m)
        if prev_zone == "priority" and cand_zone == "priority":
            score += 0.5
        elif prev_zone == "priority" and cand_zone != "priority":
            score -= 0.3

        if score > best_score:
            best_score = score
            best = (m, j, d_ratio, cand_zone)

    if best is None:
        # no candidate accepted → treat as missing
        return prev_m, prev_zone, sticky, zone_sticky, True

    m, j, d_ratio, cand_zone = best
    keep_like   = (j >= IOU_KEEP) or (d_ratio <= DIST_KEEP_RATIO)
    switch_hard = (j <= IOU_SWITCH) and (d_ratio >= DIST_SWITCH_RATIO)

    if keep_like:
        sticky = min(STICKY_FRAMES, sticky + 1)
    elif switch_hard and not warmup:
        sticky = 0
    else:
        if sticky < STICKY_FRAMES:
            return (
                prev_m if prev_m is not None else m,
                prev_zone if prev_zone else cand_zone,
                sticky + 1,
                zone_sticky,
                True
            )
        sticky = 0

    # priority zone hysteresis (stay in priority unless strongly forced out)
    if prev_zone == "priority" and cand_zone != "priority" and not warmup:
        if zone_sticky < ZONE_STICKY_FRAMES and not switch_hard:
            return (
                prev_m if prev_m is not None else m,
                prev_zone,
                sticky,
                zone_sticky + 1,
                True
            )
        zone_sticky = 0
    else:
        if cand_zone == "priority":
            zone_sticky = min(ZONE_STICKY_FRAMES, zone_sticky + 1)

    return m, cand_zone, sticky, zone_sticky, False

# =========================
# 1D cleaning helper (clip + interpolate + small median filter)
#外れ値除去（percentile clip）
#欠損補間（interpolation）
#なめらかに（median filter）
#統計情報を返す**
# =========================
def series_clip_interp_medfilt(x, fps_video,
                               p1=1, p99=99,
                               max_gap_sec=1.0,
                               median_win_fr=2):
    """
    1D cleaning for flow series:
      - percentile clipping (p1–p99)
      - linear interpolation for gaps < max_gap_sec
      - small median filter for final smoothing
    Returns (cleaned_series, stats_dict)
    """
    x = np.asarray(x, float)
    out = x.copy()
    n = len(out)
    m = np.isfinite(out)

    # percentile-based clipping
    if m.sum() >= 10:
        lo, hi = np.nanpercentile(out[m], [p1, p99])
        out[(out < lo) | (out > hi)] = np.nan

    max_gap = max(1, int(round(fps_video * max_gap_sec)))
    i = 0
    n_interp = 0

    # linear interpolation of short NaN gaps
    while i < n:
        if not np.isfinite(out[i]):
            j = i
            while j < n and not np.isfinite(out[j]):
                j += 1
            gap = j - i
            if (
                gap <= max_gap and i > 0 and j < n
                and np.isfinite(out[i - 1]) and np.isfinite(out[j])
            ):
                out[i:j] = np.linspace(out[i - 1], out[j], gap + 2)[1:-1]
                n_interp += gap
            i = j
        else:
            i += 1

    n_nan_after = np.count_nonzero(~np.isfinite(out))

    # very small median filter (optional) 3点窓
    if median_win_fr >= 2 and n >= 3:
        med = out.copy()
        med[1:-1] = np.nanmedian(
            np.vstack([out[:-2], out[1:-1], out[2:]]),
            axis=0
        )
        out = med

    stats = dict(
        total=int(n),
        nan_before=int(np.count_nonzero(~np.isfinite(x))),
        nan_after=int(n_nan_after),
        n_interp=int(n_interp),
        excl_rate=float(n_nan_after / max(n, 1)),
    )
    return out, stats

# =========================
# Main loop: segmentation + flow + split upper/lower
#1.	YOLO segmentation で「人物シルエット候補」を抽出
#2.	Skeleton（肩・股関節）と Gate（priority/allow/exclude）を使って「本物の人」だけを選択
#3.	選ばれた人物の上半身・下半身ごとに光フロー（運動量）を計算して記録

# =========================
frames = []
t_secs = []
flow_prev_gray = None

flow_all_raw   = []
flow_upper_raw = []
flow_lower_raw = []
all_vx_raw     = []
all_vy_raw     = []
upper_vx_raw   = []
upper_vy_raw   = []
lower_vx_raw   = []
lower_vy_raw   = []

prev_mask   = None
prev_zone   = "none"
sticky      = 0
zone_sticky = 0
miss_run    = 0
MISS_MAX    = int(round(MISS_MAX_SEC * fps))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # time in seconds (VFR-safe if USE_POS_MSEC=True)
    if USE_POS_MSEC:
        tm = cap.get(cv2.CAP_PROP_POS_MSEC)
        t_sec = (tm / 1000.0) if (tm and tm > 0) else (frame_idx / fps)
    else:
        t_sec = frame_idx / fps

    # skeleton index via time alignment (+ optional shift)
    t_anchor = t_sec + ANCHOR_SHIFT_SEC
    ai = int(np.clip(
        np.searchsorted(time_all, t_anchor, side='right') - 1,
        0,
        len(time_all) - 1
    ))

    # torso capsule from shoulders → hips
    pose_roi = None
    if (
        np.all(np.isfinite(shoulder_mid[ai])) and
        np.all(np.isfinite(hip_mid[ai]))
    ):
        pose_roi = capsule_mask_from_pose(
            W, H,
            (float(shoulder_mid[ai, 0]), float(shoulder_mid[ai, 1])),
            (float(hip_mid[ai, 0]), float(hip_mid[ai, 1])),
            shoulder_w_med, hip_w_med,
            CAPSULE_RADIUS_SCALE,
        )

    # anchor = hip midpoint (for anchor-based scoring)
    anchor_xy = (
        float(hip_mid[ai, 0]),
        float(hip_mid[ai, 1]),
    ) if np.all(np.isfinite(hip_mid[ai])) else None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # segmentation inference
    res = seg_model.predict(frame, verbose=False, conf=CONF_MIN)[0]
    masks_raw = (
        res.masks.data.cpu().numpy()
        if (hasattr(res, 'masks') and res.masks is not None)
        else None
    )

    if masks_raw is not None and masks_raw.shape[1:] != (H, W):
        masks = np.stack([
            cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            for m in masks_raw
        ])
    else:
        masks = masks_raw.astype(bool) if masks_raw is not None else None

    # draw zone outlines
    if SHOW_ZONES:
        draw_poly_mask(frame, priority_mask, COLOR_ZONE_PRIO, 2)
        draw_poly_mask(frame, allow_mask,    COLOR_ZONE_ALLOW, 2)
        draw_poly_mask(frame, exclude_mask,  COLOR_ZONE_EXCL,  2)

    # optional: all contours for debugging
    if DRAW_ALL_CONTOURS and (masks is not None):
        for i in range(masks.shape[0]):
            draw_cnt(frame, masks[i], COLOR_CNT, 1)

    # morphological clean-up of each candidate mask
    cand = []
    if masks is not None:
        for i in range(masks.shape[0]):
            m = (masks[i].astype(np.uint8) * 255)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k_morph, iterations=1) > 0
            m = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k_morph, iterations=1) > 0
            cand.append(m.astype(bool))

    # select one person mask (with ROI + zone logic)
    sel_all = None
    as_missing = True
    if cand:
        sel_all, prev_zone, sticky, zone_sticky, as_missing = select_mask(
            prev_mask, prev_zone, cand, t_sec, anchor_xy, sticky, zone_sticky, pose_roi
        )
        if sel_all is not None:
            draw_cnt(frame, sel_all, COLOR_SEL, 2)

    miss_run = miss_run + 1 if as_missing else 0

    # split upper vs lower body by hip line
    sel_upper = None
    sel_lower = None
    if (
        sel_all is not None and
        np.all(np.isfinite(LH[ai])) and
        np.all(np.isfinite(RH[ai]))
    ):
        lh, rh = LH[ai], RH[ai]
        mid = (lh + rh) / 2.0
        vec = rh - lh
        nrm = np.linalg.norm(vec)
        if nrm > 1e-6:
            perp = np.array([-vec[1], vec[0]], float) / nrm
            side = (XX - mid[0]) * perp[0] + (YY - mid[1]) * perp[1]
            sel_upper = sel_all & (side <= 0)
            sel_lower = sel_all & (side > 0)
            v = vec / nrm
            p1 = (int(mid[0] - v[0] * 1000), int(mid[1] - v[1] * 1000))
            p2 = (int(mid[0] + v[0] * 1000), int(mid[1] + v[1] * 1000))
            cv2.line(frame, p1, p2, COLOR_SPLIT, 2)
            cv2.circle(frame, (int(mid[0]), int(mid[1])), 4, COLOR_SPLIT, -1)

    # overlay upper / lower masks
    if sel_upper is not None:
        overlay(frame, sel_upper, COLOR_UPPER_FILL, ALPHA_UP)
        draw_cnt(frame, sel_upper, (255, 255, 255), 2)
    if sel_lower is not None:
        overlay(frame, sel_lower, COLOR_LOWER_FILL, ALPHA_LOW)
        draw_cnt(frame, sel_lower, (255, 255, 255), 2)

    # optical flow (Farneback)
    #fa = 全身の運動量
	#fu = 上半身の運動量
	#fl = 下半身の運動量
    fa = fu = fl = np.nan
    ux_all = uy_all = np.nan
    ux_u = uy_u = np.nan
    ux_l = uy_l = np.nan

    if flow_prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(
            flow_prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        fx, fy = flow[..., 0], flow[..., 1]
        mag = cv2.magnitude(fx, fy)

        if SHOW_HEAT and (sel_all is not None):
            heat_overlay(frame, mag, sel_all, ALPHA_HEAT, HEAT_PCTL)

        if sel_all is not None and sel_all.any():
            fa     = float(np.nanmean(mag[sel_all]))
            ux_all = float(np.nanmean(fx[sel_all]))
            uy_all = float(np.nanmean(fy[sel_all]))

        if sel_upper is not None and sel_upper.any():
            fu   = float(np.nanmean(mag[sel_upper]))
            ux_u = float(np.nanmean(fx[sel_upper]))
            uy_u = float(np.nanmean(fy[sel_upper]))

        if sel_lower is not None and sel_lower.any():
            fl   = float(np.nanmean(mag[sel_lower]))
            ux_l = float(np.nanmean(fx[sel_lower]))
            uy_l = float(np.nanmean(fy[sel_lower]))

        # normalize by shoulder/hip width; Y axis: upward = positive
        if np.isfinite(fa):
            fa /= shoulder_w_med
        if np.isfinite(fu):
            fu /= shoulder_w_med
        if np.isfinite(fl):
            fl /= hip_w_med

        if np.isfinite(ux_all):
            ux_all /= shoulder_w_med
        if np.isfinite(uy_all):
            uy_all = -uy_all / shoulder_w_med

        if np.isfinite(ux_u):
            ux_u /= shoulder_w_med
        if np.isfinite(uy_u):
            uy_u = -uy_u / shoulder_w_med

        if np.isfinite(ux_l):
            ux_l /= hip_w_med
        if np.isfinite(uy_l):
            uy_l = -uy_l / hip_w_med

    flow_prev_gray = gray.copy()

    frames.append(frame_idx)
    t_secs.append(t_sec)
    flow_all_raw.append(fa)
    flow_upper_raw.append(fu)
    flow_lower_raw.append(fl)
    all_vx_raw.append(ux_all)
    all_vy_raw.append(uy_all)
    upper_vx_raw.append(ux_u)
    upper_vy_raw.append(uy_u)
    lower_vx_raw.append(ux_l)
    lower_vy_raw.append(uy_l)

    out.write(frame)
    frame_idx += 1
    if sel_all is not None:
        prev_mask = sel_all

cap.release()
out.release()

# =========================
# Post-processing (clean flow series)
# =========================
df = pd.DataFrame({
    "frame": frames,
    "t_sec": t_secs,
    "fps":   fps,
    "flow_all_raw":   flow_all_raw,
    "flow_upper_raw": flow_upper_raw,
    "flow_lower_raw": flow_lower_raw,
    "all_vx_raw":     all_vx_raw,
    "all_vy_raw":     all_vy_raw,
    "upper_vx_raw":   upper_vx_raw,
    "upper_vy_raw":   upper_vy_raw,
    "lower_vx_raw":   lower_vx_raw,
    "lower_vy_raw":   lower_vy_raw,
})

mask_t0 = (df["t_sec"] >= T0_SEC)
stats = []

for key in [
    "flow_all_raw",
    "flow_upper_raw",
    "flow_lower_raw",
    "all_vx_raw",
    "all_vy_raw",
    "upper_vx_raw",
    "upper_vy_raw",
    "lower_vx_raw",
    "lower_vy_raw",
]:
    x = df[key].to_numpy()
    idx = np.where(mask_t0)[0]
    if idx.size > 0:
        clean, st = series_clip_interp_medfilt(
            x[idx],
            fps,
            p1=PCLIP_LOW,
            p99=PCLIP_HIGH,
            max_gap_sec=MISS_MAX_SEC,
            median_win_fr=MEDIAN_WIN_FR,
        )
        y = x.copy()
        y[idx] = clean
        clean_key = key.replace("_raw", "_clean")
        df[clean_key] = y
        st["series"] = clean_key
        stats.append(st)

df.to_csv(csv_path, index=False)
pd.DataFrame(stats).to_csv(stats_csv, index=False)

# Convert overlay to QuickTime-safe video
os.system(
    f"ffmpeg -y -i {out_try_path} -vcodec libx264 -pix_fmt yuv420p "
    f"-profile:v baseline -level 3.0 -movflags +faststart {out_qt_path}"
)

print("[OK] Video:", out_qt_path)
print("[OK] CSV:", csv_path)
print("[OK] Stats:", stats_csv)
