# ==========================================================
# Input  : person1_lock_track.npz  (raw keypoints)
# Output : person1_clean_kpts.npz (cleaned keypoints)
#          person1_clean_flags.json
#          person1_clean_report.json
#          person1_clean_overlay_qt.mp4
#
# Method:
# - Detect "jumps" using 1–99 percentile of velocity (dn)
# - Mark jump frames as NaN (no p-clipping)
# - Linear interpolation for short gaps (<1 s)
# - Moving-average smoothing (real-valued window)
# - Validate 4 major joints (LS, RS, LH, RH)
# - Compute shoulder/hip median width for normalization
# - Create visual overlay (red=jump, yellow=interpolated, purple=missing)
#
# Designed for GitHub publication — fully English, self-contained.
# ==========================================================

import os, json
import numpy as np
import cv2
from scipy.ndimage import uniform_filter1d

# ===============================
# I/O Paths
# ===============================
npz_in       = "/content/person1_lock_track.npz"  # ← upper-stream NPZ (no CSV)
cleaned_npz  = "/content/person1_clean_kpts.npz"
flags_json   = "/content/person1_clean_flags.json"
report_json  = "/content/person1_clean_report.json"
best_json    = "/content/bed_best.json"           # optional metadata (missing segments)
video_in     = "/content/person1_lock_track_qt.mp4"
overlay_out  = "/content/person1_clean_overlay.mp4"
overlay_qt   = "/content/person1_clean_overlay_qt.mp4"

# ===============================
# Parameters
# ===============================
P_LOW, P_HIGH    = 1.0, 99.0
INTERP_LIMIT_SEC = 1.0
VALID_MIN        = 0.90
SMOOTH_FRAMES    = 2.0
MANUAL_EXCLUDE   = False

# ===============================
# Utility
# ===============================
def diff_norm(x):
    d = np.full_like(x, np.nan)
    d[1:] = x[1:] - x[:-1]
    return np.linalg.norm(d, axis=1)

def nan_runs(mask):
    m = mask.astype(np.int8)
    dm = np.diff(np.concatenate(([0], m, [0])))
    starts = np.where(dm == 1)[0]
    ends   = np.where(dm == -1)[0]
    return list(zip(starts, ends))

def max_nan_run(mask):
    runs = nan_runs(mask)
    return max((e - s) for s, e in runs) if runs else 0

def interp_short_gaps(arr, max_gap):
    out = arr.copy()
    N, D = arr.shape
    filled = 0
    for j in range(D):
        col = arr[:, j]
        mask_nan = ~np.isfinite(col)
        runs = nan_runs(mask_nan)

        # interpolate
        col2 = col.copy()
        idx = np.arange(N)
        valid = np.isfinite(col)
        if valid.any():
            col2 = np.interp(idx, idx[valid], col[valid])
        else:
            col2[:] = np.nan

        # restore long gaps as NaN
        for s, e in runs:
            if (e - s) > max_gap:
                col2[s:e] = np.nan
            else:
                filled += np.sum(mask_nan[s:e])

        out[:, j] = col2
    return out, filled

def ma_frac(x, win_real):
    if win_real <= 1:
        return x
    k0 = int(np.floor(win_real))
    k1 = int(np.ceil(win_real))
    if k0 == k1:
        return ma_int(x, k0)
    a = win_real - k0
    return (1-a)*ma_int(x, k0) + a*ma_int(x, k1)

def ma_int(x, k):
    if k <= 1:
        return x
    y = x.copy()
    for j in range(x.shape[1]):
        col = x[:, j]
        mask = np.isfinite(col)
        if not mask.any():
            continue

        col2 = col.copy()
        col2[~mask] = 0.0
        w = mask.astype(float)

        num = uniform_filter1d(col2, size=k, mode="nearest")
        den = uniform_filter1d(w,   size=k, mode="nearest")
        y[:, j] = num / np.maximum(den, 1e-9)
    return y

# ===============================
# Load NPZ (raw keypoints)
# ===============================
raw = np.load(npz_in, allow_pickle=True)
kpts_raw = raw["kpts"]             # shape (N,17,3)
time_all = raw["time_all"]
fps      = float(raw["fps"])

N = len(time_all)

# extract 2D xy only for key joints
def get_xy(idx): return kpts_raw[:, idx, :2].astype(float)
LS = get_xy(5); RS = get_xy(6)
LH = get_xy(11); RH = get_xy(12)
nose = get_xy(0)

arr = {"nose": nose, "LS": LS, "RS": RS, "LH": LH, "RH": RH}
names = list(arr.keys())

# ===============================
# Initial NaN rate (info only)
# ===============================
initial_nan_rate = {k: float(np.mean(~np.isfinite(v).all(axis=1))) for k, v in arr.items()}

# ===============================
# Choose reference part robustly
# ===============================
shoulder_w = np.linalg.norm(RS - LS, axis=1)
hip_w      = np.linalg.norm(RH - LH, axis=1)

def valid_ratio(v):
    return float(np.isfinite(v).mean())

def cv_robust(v):
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med)) + 1e-12
    return float(mad / (med + 1e-12))

score_sh = valid_ratio(shoulder_w) / (1 + cv_robust(shoulder_w))
score_hp = valid_ratio(hip_w)      / (1 + cv_robust(hip_w))
ref_part = "shoulder" if score_sh >= score_hp else "hip"

# ===============================
# Jump detection: 1–99% velocity
# ===============================
jump_any = np.zeros(N, dtype=bool)
jump_counts = {}
per_joint_jump = {}

for k in names:
    dn = diff_norm(arr[k])
    mask = np.isfinite(dn)
    if mask.any():
        pl, ph = np.nanpercentile(dn[mask], [P_LOW, P_HIGH])
        jump = mask & ((dn <= pl) | (dn >= ph))
    else:
        jump = np.zeros_like(dn, dtype=bool)

    per_joint_jump[k] = jump
    jump_counts[k] = int(jump.sum())
    jump_any |= jump

# mark jumps as NaN
for k in names:
    arr[k][per_joint_jump[k]] = np.nan

# ===============================
# Pre-interpolation validity
# ===============================
valid4_pre = (
    np.isfinite(LS).all(axis=1) &
    np.isfinite(RS).all(axis=1) &
    np.isfinite(LH).all(axis=1) &
    np.isfinite(RH).all(axis=1)
)
valid_rate_pre = float(valid4_pre.mean())

max_run_sec_each = {k: max_nan_run(~np.isfinite(arr[k]).all(axis=1))/fps for k in names}
max_run_sec_pre  = max(max_run_sec_each.values())

if MANUAL_EXCLUDE:
    raise RuntimeError("Manual exclusion triggered")
if max_run_sec_pre >= 3.0:
    raise RuntimeError(f"Long-gap exclusion: {max_run_sec_pre:.2f}s")

# snapshot for flags
arr_pre = {k: v.copy() for k, v in arr.items()}

# ===============================
# Interpolation (<1s)
# ===============================
limit_frames = int(np.floor(INTERP_LIMIT_SEC * fps))
filled_counts = {}
for k in names:
    arr[k], filled_counts[k] = interp_short_gaps(arr[k], limit_frames)

# ===============================
# Post-interpolation validity
# ===============================
valid4_post = (
    np.isfinite(arr["LS"]).all(axis=1) &
    np.isfinite(arr["RS"]).all(axis=1) &
    np.isfinite(arr["LH"]).all(axis=1) &
    np.isfinite(arr["RH"]).all(axis=1)
)
valid_rate_post = float(valid4_post.mean())

if valid_rate_post < VALID_MIN:
    raise RuntimeError(f"valid_rate_post < {VALID_MIN}")

# ===============================
# Smoothing
# ===============================
for k in names:
    arr[k] = ma_frac(arr[k], SMOOTH_FRAMES)

# ===============================
# Median shoulder/hip width
# ===============================
shoulder_w_clean = np.linalg.norm(arr["RS"] - arr["LS"], axis=1)
hip_w_clean      = np.linalg.norm(arr["RH"] - arr["LH"], axis=1)
shoulder_w_med   = float(np.nanmedian(shoulder_w_clean))
hip_w_med        = float(np.nanmedian(hip_w_clean))

# ===============================
# Save cleaned NPZ
# ===============================
note_method = (
    f"velocity percentile {P_LOW}-{P_HIGH}, interp<{INTERP_LIMIT_SEC}s, "
    f"valid>={VALID_MIN}, MA={SMOOTH_FRAMES}"
)

np.savez(
    npz_out,
    time_all=time_all,
    fps=float(fps),
    LS=arr["LS"], RS=arr["RS"],
    LH=arr["LH"], RH=arr["RH"],
    nose=arr["nose"],
    shoulder_w_med=shoulder_w_med,
    hip_w_med=hip_w_med,
    ref_part=ref_part,
    note_method=note_method
)
print("[OK] Saved cleaned NPZ:", cleaned_npz)

# ===============================
# Save flags.json
# ===============================
flags = {
    "jump_any": jump_any.tolist(),
    "per_joint_jump": {k: per_joint_jump[k].tolist() for k in names},
    "valid4_pre": valid4_pre.tolist(),
    "valid4_post": valid4_post.tolist(),
}
with open(flags_json, "w") as f:
    json.dump(flags, f, indent=2)
print("[OK] Flags JSON:", flags_json)

# ===============================
# Save report.json
# ===============================
summary = {
    "n_frames": N,
    "fps": fps,
    "ref_part": ref_part,
    "initial_nan_rate": initial_nan_rate,
    "jump_counts": jump_counts,
    "valid_rate_pre": valid_rate_pre,
    "valid_rate_post": valid_rate_post,
    "shoulder_w_med": shoulder_w_med,
    "hip_w_med": hip_w_med,
    "note_method": note_method,
}
with open(report_json, "w") as f:
    json.dump(summary, f, indent=2)
print("[OK] Report JSON:", report_json)




# ==========================================================
# Overlay video generation
# ==========================================================

# -----------------------
# Load cleaned npz
# -----------------------
data = np.load(cleaned_npz, allow_pickle=True)

time_all = data["time_all"]
fps      = float(data["fps"])

nose = data["nose"]
LS   = data["LS"]
RS   = data["RS"]
LH   = data["LH"]
RH   = data["RH"]

N = len(time_all)

# -----------------------
# Load missing segments (tracking-level)
# -----------------------
missing_segments = []
if os.path.exists(best_json):
    try:
        meta = json.load(open(best_json))
        if "missing_segments" in meta:
            missing_segments = [(int(a), int(b)) for a,b in meta["missing_segments"]]
    except:
        pass

missing_mask = np.zeros(N, dtype=bool)
for s, e in missing_segments:
    if 0 <= s < N and 0 <= e < N:
        missing_mask[s:e+1] = True

# -----------------------
# jump_flag / interp_flag from cleaned data
#  (stored optionally inside npz)
# -----------------------
jump_mask  = np.zeros(N, dtype=bool)
interp_mask = np.zeros(N, dtype=bool)

if "jump_any" in data:
    jump_mask = data["jump_any"].astype(bool)
if "interp_any" in data:
    interp_mask = data["interp_any"].astype(bool)

# -----------------------
# Open video
# -----------------------
cap = cv2.VideoCapture(video_in)
if not cap.isOpened():
    raise RuntimeError("Cannot open video: " + video_in)

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(overlay_out, fourcc, fps, (W, H))

FONT = cv2.FONT_HERSHEY_SIMPLEX

def draw_clean_skel(vis, f):
    pts = {
        "nose": nose[f],
        "LS": LS[f],
        "RS": RS[f],
        "LH": LH[f],
        "RH": RH[f],
    }
    def ok(p):
        return np.isfinite(p).all()

    # points
    for p in pts.values():
        if ok(p):
            cv2.circle(vis, (int(p[0]), int(p[1])), 3, (0,255,255), -1)

    # shoulder line
    if ok(pts["LS"]) and ok(pts["RS"]):
        cv2.line(vis, (int(pts["LS"][0]),int(pts["LS"][1])),
                      (int(pts["RS"][0]),int(pts["RS"][1])),
                 (0,255,255), 2)

    # hip line
    if ok(pts["LH"]) and ok(pts["RH"]):
        cv2.line(vis, (int(pts["LH"][0]),int(pts["LH"][1])),
                      (int(pts["RH"][0]),int(pts["RH"][1])),
                 (0,255,255), 2)


# -----------------------
# Frame loop
# -----------------------
f = 0
while True:
    ret, frame = cap.read()
    if not ret or f >= N:
        break

    vis = frame.copy()

    # Color borders
    if missing_mask[f]:
        cv2.rectangle(vis, (0,0), (W-1,H-1), (200,0,200), 8)  # purple
    elif jump_mask[f]:
        cv2.rectangle(vis, (0,0), (W-1,H-1), (0,0,255), 8)    # red
    elif interp_mask[f]:
        cv2.rectangle(vis, (0,0), (W-1,H-1), (0,255,255), 8)  # yellow

    draw_clean_skel(vis, f)

    cv2.putText(vis, f"t={time_all[f]:.2f}s  frame={f}",
                (12,28), FONT, 0.6, (255,255,255), 2, cv2.LINE_AA)

    out.write(vis)
    f += 1

cap.release()
out.release()

# -----------------------
# Make QuickTime-safe
# -----------------------
os.system(
    f"ffmpeg -y -i {overlay_out} -vcodec libx264 -pix_fmt yuv420p "
    f"-profile:v baseline -level 3.0 -movflags +faststart {overlay_qt}"
)

print("[OK] Clean overlay video:", overlay_qt)
