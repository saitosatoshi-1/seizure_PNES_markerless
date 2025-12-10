# =============================================================
# Body-axis Optical Flow in Fixed ROI (manual polygon)
# =============================================================
import cv2, numpy as np, pandas as pd

# -----------------------------
# Paths
# -----------------------------
video_path = '/content/FBTCS_qt.mp4'
npz_path   = '/content/person1_track.npz'

out_video  = '/content/flow.mp4'
out_csv    = '/content/flow.csv'

# -----------------------------
# Parameters
# -----------------------------
MAD_Z_THR  = 6.0
DRAW_SPIKE = True

# =============================================================
# ★ priority_coords は上流セルで作成済みとして、そのまま使う
# =============================================================
print("=== Using upstream priority_coords as ROI ===")
#ROI_POINTS = priority_coords.astype(np.float32)

ROI_POINTS = np.array([
    [        556,         164],
    [        822,         146],
    [        960,         342],
    [        650,         448]
], dtype=np.float32)

print("ROI_POINTS:\n", ROI_POINTS)

# -----------------------------
# Skeleton data
# -----------------------------
dat = np.load(npz_path, allow_pickle=True)
kpts     = dat["kpts_raw"]
time_all = dat["time_all"]
fps_npz  = float(dat["fps"])

NOSE = 0
LS   = 5
RS   = 6
LH   = 11
RH   = 12

# -----------------------------
# Video IO
# -----------------------------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or fps_npz
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    out_video,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (W, H)
)

# -----------------------------
# ROI mask
# -----------------------------
mask = np.zeros((H, W), np.uint8)
cv2.fillPoly(mask, [ROI_POINTS.astype(np.int32)], 1)
mask_bool = mask.astype(bool)

# -----------------------------
# 状態変数
# -----------------------------
flow_prev = None
rows = []
mag_hist = []
frame_idx = 0

prev_vx = None
prev_vy = None

# =============================================================
# Main loop
# =============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- skeleton index ----------
    t_sec = frame_idx / fps
    ai = int(np.clip(np.searchsorted(time_all, t_sec, 'right') - 1,
                     0, len(time_all) - 1))
    kp = kpts[ai]

    # -----------------------------
    # Body coordinate axes
    # -----------------------------
    LS_xy = kp[LS, :2]
    RS_xy = kp[RS, :2]
    LH_xy = kp[LH, :2]
    RH_xy = kp[RH, :2]

    axes_valid = False

    if (np.all(np.isfinite(LS_xy)) and np.all(np.isfinite(RS_xy)) and
        np.all(np.isfinite(LH_xy)) and np.all(np.isfinite(RH_xy))):

        shoulder_mid = (LS_xy + RS_xy) / 2.0
        hip_mid      = (LH_xy + RH_xy) / 2.0

        vx = RS_xy - LS_xy
        vy = hip_mid - shoulder_mid

        nx = np.linalg.norm(vx)
        ny = np.linalg.norm(vy)

        if nx > 1e-6 and ny > 1e-6:
            vx = vx / nx
            vy = vy / ny
            axes_valid = True
            prev_vx, prev_vy = vx, vy

    else:
        if prev_vx is not None:
            vx, vy = prev_vx, prev_vy
            axes_valid = True

    # -----------------------------
    # Optical flow
    # -----------------------------
    if flow_prev is not None:
        flow = cv2.calcOpticalFlowFarneback(
            flow_prev, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        fx = flow[...,0]
        fy = flow[...,1]

        if axes_valid:
            vx_x, vx_y = vx
            vy_x, vy_y = vy

            fx_body = fx * vx_x + fy * vx_y
            fy_body = fx * vy_x + fy * vy_y

            mag_body = cv2.magnitude(fx_body, fy_body)

            vx_mean  = float(np.mean(fx_body[mask_bool]))
            vy_mean  = float(np.mean(fy_body[mask_bool]))
            mag_mean = float(np.mean(mag_body[mask_bool]))

            # カラーマップ方向
            ang_body = np.arctan2(fy_body, fx_body)

            hsv = np.zeros((H,W,3), np.uint8)
            hsv[...,1] = 255
            hue = (ang_body + np.pi) / (2*np.pi) * 180.0
            hsv[...,0] = hue.astype(np.uint8)
            hsv[...,2] = cv2.normalize(mag_body, None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)

            flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            masked_flow = np.zeros_like(flow_color)
            masked_flow[mask_bool] = flow_color[mask_bool]
            overlay = cv2.addWeighted(frame,0.35,masked_flow,0.65,0)

        else:
            vx_mean = vy_mean = mag_mean = np.nan
            overlay = frame.copy()

    else:
        vx_mean = vy_mean = mag_mean = np.nan
        overlay = frame.copy()

    flow_prev = gray.copy()

    # -----------------------------
    # MAD spike detection
    # -----------------------------
    spike_flag = 0
    mag_hist.append(mag_mean)

    arr = np.array([x for x in mag_hist if np.isfinite(x)])
    if len(arr) >= 20:
        med = np.median(arr)
        mad = np.median(np.abs(arr - med)) + 1e-9
        z = abs(mag_mean - med) / mad if np.isfinite(mag_mean) else 0

        if np.isfinite(z) and z > MAD_Z_THR:
            spike_flag = 1
            print(f"[SPIKE] frame {frame_idx}  t={t_sec:.3f}  mag={mag_mean:.4f}  z={z:.2f}")

    color = (0,0,255) if (spike_flag and DRAW_SPIKE) else (0,255,255)
    cv2.polylines(overlay, [ROI_POINTS.astype(int)], True, color, 2)

    out.write(overlay)

    rows.append([frame_idx, t_sec, vx_mean, vy_mean, mag_mean, spike_flag])

    frame_idx += 1

cap.release()
out.release()

df = pd.DataFrame(rows, columns=["frame","t_sec","vx_body","vy_body","mag_body","spike"])
df.to_csv(out_csv, index=False)

print("=== DONE ===")
print("Video:", out_video)
print("CSV  :", out_csv)
