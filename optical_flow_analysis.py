# ==========================================================
# Skeleton + Silhouette Optical Flow (Stable person1 tracking)
# ----------------------------------------------------------
# This script performs seizure-motion quantification using:
#   (1) Skeleton keypoints (cleaned)
#   (2) Silhouette segmentation (YOLO11-seg)
#   (3) Farnebäck optical flow
#
# Key Technical Ideas
# -------------------
# - Robust "person1" tracking in clinical VEEG videos where assistants often appear.
# - Person selection uses ONLY:
#       • Priority zone from gate_config.json (bed area)
#       • Torso capsule ROI from skeleton (shoulder → hip)
#       • Hip-anchor proximity
#   → IoU-based tracking is intentionally NOT used (causes jumping to assistants).
#
# - Silhouette mask is split into upper/lower body using a line orthogonal to the
#   hip vector (same definition used in the manuscript).
#
# - Optical flow (Farnebäck) is used because:
#       • It is dense (per-pixel), robust for low-resolution clinical videos.
#       • Motion magnitude is stable even when pose estimation momentarily drifts.
#
# Output
# ------
# - Cleaned silhouette-flow time series (CSV)
# - Summary statistics (CSV)
# - Overlay visualization video (MP4)
# - Flow-guided refined keypoints (NPZ)
# - Per-frame JSON log for debugging / reproducibility
# ==========================================================

import os, json
import numpy as np
import cv2
from ultralytics import YOLO
import pandas as pd

# =========================
# I/O paths
# =========================
video_path   = '/content/FBTCS_silhouette_qt.mp4'
npz_path     = '/content/person1_clean_kpts.npz'
gate_json    = '/content/gate_config.json'

csv_path     = '/content/person1_silhouette_flow.csv'
stats_csv    = '/content/person1_flow_stats.csv'

out_try_path = '/content/person1_silhouette_seg_clean.mp4'
out_qt_path  = '/content/person1_silhouette_seg_clean_qt.mp4'

# 追加：flow-guided refined keypoints / JSON log の出力先
refined_npz_path = '/content/person1_refined_kpts.npz'
json_log_path    = '/content/person1_flow_pose_log.json'

# =========================
# Visualization colors
# =========================
COLOR_UPPER_FILL = (60, 200, 60)
COLOR_LOWER_FILL = (200, 60, 200)
COLOR_SPLIT      = (0, 255, 255)
COLOR_SEL        = (80, 255, 80)
COLOR_CNT        = (170, 170, 170)

COLOR_ZONE_PRIO  = (0, 200, 255)
COLOR_ZONE_ALLOW = (120, 120, 255)
COLOR_ZONE_EXCL  = (60, 60, 60)

ALPHA_UP   = 0.40
ALPHA_LOW  = 0.40
ALPHA_HEAT = 0.30
HEAT_PCTL  = 95

DRAW_ALL_CONTOURS = True
SHOW_HEAT         = True
SHOW_ZONES        = True

# =========================
# Hyperparameters
# =========================
CONF_MIN        = 0.25
KERNEL_MORPH    = (3, 3)

T0_SEC          = 0.0
MISS_MAX_SEC    = 1.0
MEDIAN_WIN_FR   = 2
PCLIP_LOW       = 1
PCLIP_HIGH      = 99

USE_POS_MSEC    = True
ANCHOR_SHIFT_SEC = 0.0

LOCK_WARMUP_SEC      = 3.0
ANCHOR_RADIUS_RATIO  = 0.10
ANCHOR_CENT_MAX_RATIO = 0.12

MIN_AREA_RATIO   = 0.005
MIN_HEIGHT_RATIO = 0.15
MAX_ASPECT       = 6.0

POSE_OVERLAP_MIN    = 0.25
POSE_BONUS          = 1.5
CAPSULE_RADIUS_SCALE = 0.9

# Flow-guided keypoint smoothing weight (0.3 推奨)
FLOW_KP_ALPHA = 0.3

# =========================
# Segmentation model
# =========================
seg_model = YOLO('yolo11x-seg.pt')

# =========================
# Load video
# =========================
cap = cv2.VideoCapture(video_path)
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
# Load cleaned skeleton
# =========================
dat = np.load(npz_path)
time_all = dat["time_all"]
LS, RS   = dat["LS"], dat["RS"]
LH, RH   = dat["LH"], dat["RH"]

has_refW = "frame_w" in dat.files
has_refH = "frame_h" in dat.files

if has_refW and has_refH:
    refW = dat["frame_w"].item()
    refH = dat["frame_h"].item()
else:
    refW, refH = W, H

hip_mid      = (LH + RH) / 2
shoulder_mid = (LS + RS) / 2
shoulder_w_med = float(np.nanmedian(np.linalg.norm(RS-LS, axis=1)))
hip_w_med      = float(np.nanmedian(np.linalg.norm(RH-LH, axis=1)))

# =========================
# Load gate masks
# =========================
def _as_polys(obj):
    if obj is None:
        return []
    if isinstance(obj, dict):
        for k in ("priority","allow","exclude","bed_poly"):
            if k in obj:
                obj = obj[k]
                break
    if isinstance(obj, list):
        if len(obj) > 0 and isinstance(obj[0][0], (list, tuple)):
            return [np.array(p, float) for p in obj]
        return [np.array(obj, float)]
    return []


def load_gate_masks(json_path, W, H):
    """
    Load priority/allow/exclude zones from gate_config.json and convert them
    to boolean masks.
    - priority:     strong constraint for selecting person1 inside the bed area
    - allow:        softer constraint (outside bed but acceptable)
    - exclude:      masks to completely ignore (e.g., equipment, assistants)
    """
    if not os.path.isfile(json_path):
        return None, None, None

    js = json.load(open(json_path, 'r'))

    # polygon listを作成
    pri_list   = _as_polys(js.get("priority"))
    allow_list = _as_polys(js.get("allow"))
    excl_list  = _as_polys(js.get("exclude"))

    # 画像サイズの2Dマスクをpolygon塗りつぶしで作る
    pri = np.zeros((H, W), np.uint8)
    for P in pri_list:
        cv2.fillPoly(pri, [P.astype(np.int32)], 1)

    allow = np.ones((H, W), np.uint8) if not allow_list else np.zeros((H, W), np.uint8)
    for P in allow_list:
        cv2.fillPoly(allow, [P.astype(np.int32)], 1)

    excl = np.zeros((H, W), np.uint8)
    for P in excl_list:
        cv2.fillPoly(excl, [P.astype(np.int32)], 1)

    return pri.astype(bool), allow.astype(bool), excl.astype(bool)

# priority_mask  : ベッド内（最優先）
# allow_mask     : ベッド周囲（許容）
# exclude_mask   : 完全に無視する領域
priority_mask, allow_mask, exclude_mask = load_gate_masks(gate_json, W, H)


# =========================
# Utilities
# =========================
# マスクの中心 (重心) を求める関数
def mcent(m):
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


# 体幹のカプセルROIを作る
def capsule_mask_from_pose(W, H, shoulder_xy, hip_xy, sw, hw, scale):
    """
    Generate an approximate torso mask ("capsule") using two circles connected
    by a thick line:
        - upper circle: around shoulders
        - lower circle: around hips
    This ROI drastically stabilizes silhouette tracking by removing
    non-torso candidates such as assistants, blankets, or noise.
    """
    canvas = np.zeros((H, W), np.uint8)
    p1, p2 = np.array(shoulder_xy), np.array(hip_xy)
    if not (np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))):
        return canvas.astype(bool)

    r1 = max(6.0, sw * scale)
    r2 = max(6.0, hw * scale)

    cv2.circle(canvas, tuple(p1.astype(int)), int(r1), 255, -1)
    cv2.circle(canvas, tuple(p2.astype(int)), int(r2), 255, -1)
    thickness = int(r1 + r2)
    cv2.line(canvas, tuple(p1.astype(int)), tuple(p2.astype(int)), 255, thickness)

    M = (canvas > 0)
    return M & allow_mask if allow_mask is not None else M


# 人以外のものをフィルタする, 面積が小さすぎないか, 縦横比が異常じゃないか, 高さが最低限あるか
def is_ok_shape(m):
    area = m.sum()
    if area < MIN_AREA_RATIO * (W * H):
        return False
    ys, xs = np.nonzero(m)
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    if max(w / h, h / w) > MAX_ASPECT:
        return False
    if h < MIN_HEIGHT_RATIO * H:
        return False
    return True


# 半透明でマスク表示
def overlay_color(img, m, col, a):
    if m is None or not m.any():
        return
    ov = img.copy()
    ov[m] = (ov[m] * (1 - a) + np.array(col) * a).astype(np.uint8)
    img[:] = ov


# optical flowの速度magnitudeを0-255に正規化
# 赤は強い動き
def heat_overlay(img, mag, m):
    if m is None or not m.any():
        return
    vmax = np.nanpercentile(mag[m], HEAT_PCTL)
    vmax = max(vmax, 1e-6)
    heat = (np.clip(mag / vmax, 0, 1) * 255).astype(np.uint8)
    cmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    ov = img.copy()
    ov[m] = (cmap[m] * ALPHA_HEAT + img[m] * (1 - ALPHA_HEAT)).astype(np.uint8)
    img[:] = ov


# マスクがpriority, allow, noneのどこに属しているか判定
def zone_of_mask(m):
    if priority_mask is not None and (m & priority_mask).any():
        return "priority"
    if allow_mask is not None and (m & allow_mask).any():
        return "allow"
    return "none"


# 前のフレームとの重なりをみて, 同じ人物かチェック
def iou(a, b):
    """IoU of two boolean masks."""
    if a is None or b is None:
        return 0.0
    inter = np.logical_and(a, b).sum()
    uni   = np.logical_or(a, b).sum()
    return inter / (uni + 1e-6)


def refine_kp_with_flow(raw_xy, prev_xy, flow):
    """
    Flow-guided smoothing of a single keypoint.
    raw_xy : [x, y] from current skeleton
    prev_xy: [x, y] refined from previous frame
    flow   : H x W x 2 Farneback flow (prev -> current)
    """
    if raw_xy is None:
        return raw_xy
    raw = np.asarray(raw_xy, float)
    if not np.all(np.isfinite(raw)):
        return raw
    if prev_xy is None:
        return raw
    prev = np.asarray(prev_xy, float)
    if not np.all(np.isfinite(prev)):
        return raw

    x_prev = int(round(prev[0]))
    y_prev = int(round(prev[1]))
    if x_prev < 0 or x_prev >= W or y_prev < 0 or y_prev >= H:
        return raw

    vx = float(flow[y_prev, x_prev, 0])
    vy = float(flow[y_prev, x_prev, 1])
    pred = prev + np.array([vx, vy], float)
    if not np.all(np.isfinite(pred)):
        return raw

    return (1.0 - FLOW_KP_ALPHA) * raw + FLOW_KP_ALPHA * pred


def xy_to_list(xy):
    """Convert [x,y] (with possible NaN) to JSON-friendly [float or None, float or None]."""
    if xy is None:
        return [None, None]
    xy = np.asarray(xy, float)
    out = []
    for v in xy:
        out.append(float(v) if np.isfinite(v) else None)
    return out


# ==========================================================
# ★ 最重要：person1 を選択するシンプル select_mask
# ==========================================================
def select_mask(prev_m, prev_zone, cand_list, t_sec, anchor_xy, sticky, zone_sticky, pose_roi):
    """
    Select the true "person1" silhouette from multiple segmentation candidates.
    Core selection logic:
        1. Candidate must lie inside priority/allow zones.
        2. Candidate must overlap with torso ROI (capsule from skeleton).
        3. Candidate close to hip-anchor gets higher score.
        4. IoU with previous mask is used *lightly* (not as hard tracking).
    This is intentionally simple and deterministic for clinical use.
    Returns:
        best_mask, best_zone, sticky, zone_sticky, is_missing
    """

    # pose ROI がなければ何もできない
    pose_valid = (pose_roi is not None) and pose_roi.any()
    print(f"[DBG] t={t_sec:.2f}, N_cand={len(cand_list)}, pose_roi_valid={pose_valid}")
    if not pose_valid:
        return prev_m, prev_zone, sticky, zone_sticky, True  # missing 扱い

    diag = np.hypot(W, H)
    warmup = (t_sec <= LOCK_WARMUP_SEC) and (anchor_xy is not None)
    roi_area = float(pose_roi.sum()) + 1e-6

    # 前フレーム中心（IoU／距離用）
    prev_c = mcent(prev_m) if prev_m is not None else None

    best_mask = None
    best_zone = "none"
    best_score = 0.0

    for idx, m in enumerate(cand_list):
        # ---- 基本チェック ----
        if m is None or not m.any():
            print(f"[DBG]  cand[{idx}] empty → skip")
            continue
        if exclude_mask is not None and (m & exclude_mask).any():
            print(f"[DBG]  cand[{idx}] in exclude → skip")
            continue
        if not is_ok_shape(m):
            print(f"[DBG]  cand[{idx}] shape NG → skip")
            continue

        # ゾーン判定
        z = zone_of_mask(m)
        print(f"[DBG]  cand[{idx}] zone={z}")
        if z == "none":
            print(f"[DBG]  cand[{idx}] zone none → skip")
            continue

        # pose ROI との重なり
        inter = np.logical_and(pose_roi, m).sum()
        if inter == 0:
            print(f"[DBG]  cand[{idx}] no ROI overlap → skip")
            continue

        cand_area = float(m.sum()) + 1e-6
        o_roi  = inter / roi_area       # ROI側からみた埋まり具合
        o_cand = inter / cand_area      # 候補側からみた「どれだけ胴体に一致」

        # anchor(hip_mid) との距離 → 近いほど高スコア
        anchor_term = 0.0
        if anchor_xy is not None:
            c_now = mcent(m)
            if c_now is not None:
                d = np.linalg.norm(np.array(c_now) - np.array(anchor_xy)) / (diag + 1e-6)
                anchor_term = 1.0 - min(1.0, d)

        # warmup 中は ROI をある程度埋めていないと弾く
        if warmup and o_roi < 0.40:
            print(f"[DBG]  cand[{idx}] warmup & o_roi={o_roi:.3f} < 0.40 → skip")
            continue

        # 前フレームとの IoU（軽くスコアに加える程度）
        j = iou(prev_m, m) if prev_m is not None else 0.0
        if prev_c is None:
            d_ratio = 1.0
        else:
            c_prev = np.array(prev_c)
            c_now = mcent(m)
            if c_now is None:
                d_ratio = 1.0
            else:
                d_ratio = np.linalg.norm(np.array(c_now) - c_prev) / (diag + 1e-6)

        # ---- スコア計算 ----
        score = (
            1.5 * o_roi +          # 胴体 ROI をどれだけ埋めるか（最重要）
            0.5 * o_cand +         # 「候補のどれだけが胴体か」
            0.5 * anchor_term +    # anchor への近さ
            0.3 * j                # 前フレームとの IoU（連続性）
        )

        # ゾーンボーナス：priority > allow
        if z == "priority":
            score += 0.3
        elif z == "allow":
            score += 0.1

        print(
            f"[DBG]  cand[{idx}] "
            f"o_roi={o_roi:.3f}, o_cand={o_cand:.3f}, "
            f"anchor={anchor_term:.3f}, IoU={j:.3f}, score={score:.3f}"
        )

        if score > best_score:
            best_score = score
            best_mask = m
            best_zone = z

    # ---- 最終判定 ----
    MIN_SCORE = 0.35
    if best_mask is None or best_score < MIN_SCORE:
        print(f"[DBG] reject: best_score={best_score:.3f} < {MIN_SCORE}")
        return prev_m, prev_zone, sticky, zone_sticky, True  # missing

    print(f"[DBG] accept: zone={best_zone}, score={best_score:.3f}")
    # ここでは sticky / zone_sticky は触らずそのまま返す（シンプル運用）
    return best_mask, best_zone, sticky, zone_sticky, False


# =========================
# Flow computing
# =========================
frames   = []
t_secs   = []
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

# 追加：raw / refined keypoints（動画フレームごと）
LS_raw_list = []
RS_raw_list = []
LH_raw_list = []
RH_raw_list = []

LS_ref_list = []
RS_ref_list = []
LH_ref_list = []
RH_ref_list = []

# 前フレームの refined keypoints（flow補正用）
prev_LS_ref = None
prev_RS_ref = None
prev_LH_ref = None
prev_RH_ref = None

# JSON logging 用
json_logs = []

prev_mask = None
prev_zone = "none"

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if USE_POS_MSEC:
        tm = cap.get(cv2.CAP_PROP_POS_MSEC)
        t_sec = (tm / 1000.0) if (tm and tm > 0) else (frame_idx / fps)
    else:
        t_sec = frame_idx / fps

    # skeleton alignment
    t_anchor = t_sec + ANCHOR_SHIFT_SEC
    ai = int(np.clip(np.searchsorted(time_all, t_anchor, 'right') - 1, 0, len(time_all) - 1))

    # current raw keypoints from skeleton
    ls_raw = LS[ai].copy()
    rs_raw = RS[ai].copy()
    lh_raw = LH[ai].copy()
    rh_raw = RH[ai].copy()

    # 初期値として refined = raw
    ls_ref = ls_raw.copy()
    rs_ref = rs_raw.copy()
    lh_ref = lh_raw.copy()
    rh_ref = rh_raw.copy()

    pose_roi = capsule_mask_from_pose(
        W, H,
        shoulder_mid[ai],
        hip_mid[ai],
        shoulder_w_med,
        hip_w_med,
        CAPSULE_RADIUS_SCALE
    )
    anchor_xy = tuple(hip_mid[ai]) if np.all(np.isfinite(hip_mid[ai])) else None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # segmentation
    res = seg_model.predict(frame, verbose=False, conf=CONF_MIN)[0]
    masks_raw = (
        res.masks.data.cpu().numpy()
        if (hasattr(res, 'masks') and res.masks is not None)
        else None
    )
    print("masks_raw:", None if masks_raw is None else masks_raw.shape)

    if masks_raw is not None and masks_raw.shape[1:] != (H, W):
        masks = np.stack(
            [cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0 for m in masks_raw]
        )
    else:
        masks = masks_raw.astype(bool) if masks_raw is not None else None

    if SHOW_ZONES and priority_mask is not None:
        cnt, _ = cv2.findContours(priority_mask.astype(np.uint8) * 255,
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, cnt, -1, COLOR_ZONE_PRIO, 2)

    cand = []
    if masks is not None:
        for m in masks:
            m2 = cv2.morphologyEx((m * 255).astype(np.uint8), cv2.MORPH_OPEN, k_morph) > 0
            m2 = cv2.morphologyEx((m2 * 255).astype(np.uint8), cv2.MORPH_CLOSE, k_morph) > 0
            cand.append(m2.astype(bool))

    sel_all, prev_zone, _, _, as_missing = select_mask(
        prev_mask, prev_zone, cand, t_sec, anchor_xy, 0, 0, pose_roi
    )

    if sel_all is not None:
        prev_mask = sel_all
        cnt, _ = cv2.findContours(
            (sel_all * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(frame, cnt, -1, COLOR_SEL, 2)

    # upper / lower split
    # Split silhouette into upper/lower parts using a line orthogonal to
    # the left-hip → right-hip vector.
    sel_upper = sel_lower = None
    if sel_all is not None:
        lh, rh = LH[ai], RH[ai]
        if np.all(np.isfinite(lh)) and np.all(np.isfinite(rh)):
            mid = (lh + rh) / 2
            vec = rh - lh
            nrm = np.linalg.norm(vec)
            if nrm > 1e-6:
                perp = np.array([-vec[1], vec[0]]) / nrm
                side = (XX - mid[0]) * perp[0] + (YY - mid[1]) * perp[1]
                sel_upper = sel_all & (side <= 0)
                sel_lower = sel_all & (side > 0)

    # Farnebäck dense optical flow:
    fa = fu = fl = np.nan
    ux_all = uy_all = ux_u = uy_u = ux_l = uy_l = np.nan

    flow = None
    if flow_prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(
            flow_prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        fx, fy = flow[..., 0], flow[..., 1]
        mag = cv2.magnitude(fx, fy)

        if SHOW_HEAT and sel_all is not None:
            heat_overlay(frame, mag, sel_all)

        if sel_all is not None and sel_all.any():
            fa = np.nanmean(mag[sel_all])
            ux_all = np.nanmean(fx[sel_all])
            uy_all = np.nanmean(fy[sel_all])

        if sel_upper is not None and sel_upper.any():
            fu = np.nanmean(mag[sel_upper])
            ux_u = np.nanmean(fx[sel_upper])
            uy_u = np.nanmean(fy[sel_upper])

        if sel_lower is not None and sel_lower.any():
            fl = np.nanmean(mag[sel_lower])
            ux_l = np.nanmean(fx[sel_lower])
            uy_l = np.nanmean(fy[sel_lower])

        # normalize
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

        # === ここで keypoints を flow で補正（refined） ===
        ls_ref = refine_kp_with_flow(ls_raw, prev_LS_ref, flow)
        rs_ref = refine_kp_with_flow(rs_raw, prev_RS_ref, flow)
        lh_ref = refine_kp_with_flow(lh_raw, prev_LH_ref, flow)
        rh_ref = refine_kp_with_flow(rh_raw, prev_RH_ref, flow)

    # 次フレーム用に grayscale を保存
    flow_prev_gray = gray.copy()

    # time series / flow の記録
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

    # keypoints の記録
    LS_raw_list.append(ls_raw)
    RS_raw_list.append(rs_raw)
    LH_raw_list.append(lh_raw)
    RH_raw_list.append(rh_raw)

    LS_ref_list.append(ls_ref)
    RS_ref_list.append(rs_ref)
    LH_ref_list.append(lh_ref)
    RH_ref_list.append(rh_ref)

    # 次フレーム用の refined keypoints
    prev_LS_ref = ls_ref
    prev_RS_ref = rs_ref
    prev_LH_ref = lh_ref
    prev_RH_ref = rh_ref

    # JSON ログ（1フレーム分）
    log_entry = {
        "frame": int(frame_idx),
        "t_sec": float(t_sec),
        "ai": int(ai),
        "zone": prev_zone,
        "as_missing": bool(as_missing),
        "anchor_xy": None if anchor_xy is None else xy_to_list(anchor_xy),
        "LS_raw": xy_to_list(ls_raw),
        "RS_raw": xy_to_list(rs_raw),
        "LH_raw": xy_to_list(lh_raw),
        "RH_raw": xy_to_list(rh_raw),
        "LS_refined": xy_to_list(ls_ref),
        "RS_refined": xy_to_list(rs_ref),
        "LH_refined": xy_to_list(lh_ref),
        "RH_refined": xy_to_list(rh_ref),
        "flow_summary": {
            "fa": None if not np.isfinite(fa) else float(fa),
            "fu": None if not np.isfinite(fu) else float(fu),
            "fl": None if not np.isfinite(fl) else float(fl),
            "ux_all": None if not np.isfinite(ux_all) else float(ux_all),
            "uy_all": None if not np.isfinite(uy_all) else float(uy_all),
            "ux_upper": None if not np.isfinite(ux_u) else float(ux_u),
            "uy_upper": None if not np.isfinite(uy_u) else float(uy_u),
            "ux_lower": None if not np.isfinite(ux_l) else float(ux_l),
            "uy_lower": None if not np.isfinite(uy_l) else float(uy_l),
        }
    }
    json_logs.append(log_entry)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

# =========================
# Series cleaning
# =========================
def series_clip_interp_medfilt(x, fps, p1=1, p99=99, max_gap_sec=1.0, median_win_fr=2):
    """
    Clean a 1-D time series by:
        1. Clipping extreme values (1–99 percentile)
        2. Linear interpolation for gaps < max_gap_sec
        3. 3-point median filter to suppress noise
    Returns:
        cleaned_series, stats(dict)
    """
    x = np.asarray(x, float)
    out = x.copy()
    n = len(out)
    m = np.isfinite(out)

    if m.sum() >= 10:
        lo, hi = np.nanpercentile(out[m], [p1, p99])
        out[(out < lo) | (out > hi)] = np.nan

    max_gap = max(1, int(round(fps * max_gap_sec)))
    i = 0
    n_interp = 0
    while i < n:
        if not np.isfinite(out[i]):
            j = i
            while j < n and not np.isfinite(out[j]):
                j += 1
            gap = j - i
            if gap <= max_gap and i > 0 and j < n:
                out[i:j] = np.linspace(out[i - 1], out[j], gap + 2)[1:-1]
                n_interp += gap
            i = j
        else:
            i += 1

    n_nan_after = np.count_nonzero(~np.isfinite(out))

    if median_win_fr >= 2 and n >= 3:
        med = out.copy()
        med[1:-1] = np.nanmedian(
            np.vstack([out[:-2], out[1:-1], out[2:]]), axis=0
        )
        out = med

    return out, dict(
        total=n,
        nan_before=int(np.count_nonzero(~np.isfinite(x))),
        nan_after=int(n_nan_after),
        n_interp=int(n_interp),
        excl_rate=float(n_nan_after / max(n, 1)),
    )


df = pd.DataFrame({
    "frame": frames,
    "t_sec": t_secs,
    "fps": fps,
    "flow_all_raw": flow_all_raw,
    "flow_upper_raw": flow_upper_raw,
    "flow_lower_raw": flow_lower_raw,
    "all_vx_raw": all_vx_raw,
    "all_vy_raw": all_vy_raw,
    "upper_vx_raw": upper_vx_raw,
    "upper_vy_raw": upper_vy_raw,
    "lower_vx_raw": lower_vx_raw,
    "lower_vy_raw": lower_vy_raw,
})

mask_t0 = (df["t_sec"] >= T0_SEC)
stats = []

for key in [
    "flow_all_raw", "flow_upper_raw", "flow_lower_raw",
    "all_vx_raw", "all_vy_raw",
    "upper_vx_raw", "upper_vy_raw",
    "lower_vx_raw", "lower_vy_raw"
]:
    x = df[key].to_numpy()
    idx = np.where(mask_t0)[0]
    if idx.size > 0:
        clean, st = series_clip_interp_medfilt(
            x[idx], fps, PCLIP_LOW, PCLIP_HIGH, MISS_MAX_SEC, MEDIAN_WIN_FR
        )
        y = x.copy()
        y[idx] = clean
        df[key.replace("_raw", "_clean")] = y
        st["series"] = key.replace("_raw", "_clean")
        stats.append(st)

df.to_csv(csv_path, index=False)
pd.DataFrame(stats).to_csv(stats_csv, index=False)

# =========================
# Save refined keypoints (NPZ) & JSON log
# =========================
np.savez(
    refined_npz_path,
    frame=np.array(frames, dtype=int),
    t_sec=np.array(t_secs, dtype=float),
    fps=float(fps),
    W=int(W),
    H=int(H),
    LS_raw=np.asarray(LS_raw_list, float),
    RS_raw=np.asarray(RS_raw_list, float),
    LH_raw=np.asarray(LH_raw_list, float),
    RH_raw=np.asarray(RH_raw_list, float),
    LS_refined=np.asarray(LS_ref_list, float),
    RS_refined=np.asarray(RS_ref_list, float),
    LH_refined=np.asarray(LH_ref_list, float),
    RH_refined=np.asarray(RH_ref_list, float),
)

with open(json_log_path, 'w', encoding='utf-8') as f:
    json.dump(json_logs, f, ensure_ascii=False, indent=2)

os.system(
    f"ffmpeg -y -i {out_try_path} -vcodec libx264 -pix_fmt yuv420p "
    f"-profile:v baseline -level 3.0 -movflags +faststart {out_qt_path}"
)

print("=== DONE ===")
print("[Video]", out_qt_path)
print("[CSV]", csv_path)
print("[Stats]", stats_csv)
print("[Refined kpts NPZ]", refined_npz_path)
print("[JSON log]", json_log_path)
