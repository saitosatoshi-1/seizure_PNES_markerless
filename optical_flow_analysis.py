# =============================================================
# Body-axis Optical Flow in Fixed ROI
# -------------------------------------------------------------
# This script computes dense optical flow within a manually
# specified polygonal ROI, and projects the flow vectors onto
# body-centric axes derived from skeleton keypoints.
#
# Key idea:
#   - Use shoulder and hip landmarks to define a local
#     body coordinate system (vx = shoulder axis, vy = torso axis).
#   - Convert optical flow from image coordinates → body coordinates.
#   - This reduces camera-angle dependency and isolates true body motion.
#
# Outputs:
#   1) flow.mp4 : optical-flow overlay video
#   2) flow.csv : per-frame motion metrics (vx_body, vy_body, mag_body)
#
# Dependencies:
#   OpenCV, NumPy, pandas
#
# =============================================================

import cv2, numpy as np, pandas as pd

# =============================================================
# Paths
# =============================================================
video_path = '/content/FBTCS_qt.mp4'
npz_path   = '/content/person1_track.npz'

out_video  = '/content/flow.mp4'
out_csv    = '/content/flow.csv'

# =============================================================
# Parameters
# =============================================================
MAD_Z_THR  = 6.0     # threshold for MAD-based spike detection
DRAW_SPIKE = True     # draw ROI in red if spike detected

# =============================================================
# ROI polygon (priority zone from upstream step or manual input)
# =============================================================

ROI_POINTS = np.array([
    [556, 164],
    [822, 146],
    [960, 342],
    [650, 448]
], dtype=np.float32)


# =============================================================
# Load skeleton (person1 tracking)
#   kpts_raw : pose keypoints
#   time_all : timestamp for each frame of skeleton
# =============================================================
dat = np.load(npz_path, allow_pickle=True)
kpts     = dat["kpts_raw"]
time_all = dat["time_all"]
fps_npz  = float(dat["fps"])

# landmark indices (COCO format)
NOSE = 0
LS   = 5
RS   = 6
LH   = 11
RH   = 12

# =============================================================
# Video IO
# =============================================================
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    out_video,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (W, H)
)

# =============================================================
# ROI mask
# 中身：すべて 0
# 型：uint8（0〜255 の画素値を持つ画像として扱いやすい）
# 真っ黒（0）の画像を作る
# ROIを1でぬりつぶす (fillpoly)
# ROI部分をtrueにする
# =============================================================
mask = np.zeros((H, W), np.uint8)
cv2.fillPoly(mask, [ROI_POINTS.astype(np.int32)], 1)
mask_bool = mask.astype(bool)

# =============================================================
# State variables
# =============================================================
flow_prev = None
rows = []
mag_hist = []
frame_idx = 0

prev_vx = None
prev_vy = None

# =============================================================
# Main loop
# 動画のフレームを1枚ずつ読み込み、カラー → グレースケールに変換して処理する
# frame：読み込んだ画像（1フレーム分）
# ret：成功したかどうか（True or False）
# =============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------------
    # Match skeleton timestamp to video time
    # 動画フレームと skeleton フレームを「時刻」で同期させる処理
    # SkeletonはYOLO推論ベースなので, fps時刻とずれがある
    # そのフレームに対応する肩・股関節座標（kp）を取り出す
    # ---------------------------------------------------------
    t_sec = frame_idx / fps
    ai = int(np.clip(np.searchsorted(time_all, t_sec, 'right') - 1,
                     0, len(time_all) - 1))
    kp = kpts[ai]

    # ---------------------------------------------------------
    # Derive body coordinate axes (vx, vy)
    # vx: shoulder axis
    # vy: torso axis (shoulder_mid → hip_mid)
　　# LS: 左肩, RS: 右肩, LH: 左股関節, RH: 右股関節
    # ---------------------------------------------------------
    LS_xy = kp[LS, :2]
    RS_xy = kp[RS, :2]
    LH_xy = kp[LH, :2]
    RH_xy = kp[RH, :2]

    axes_valid = False

    # NaNが無いか確認
    if (np.all(np.isfinite(LS_xy)) and np.all(np.isfinite(RS_xy)) and
        np.all(np.isfinite(LH_xy)) and np.all(np.isfinite(RH_xy))):

        # 肩の中点・股関節の中点
        shoulder_mid = (LS_xy + RS_xy) / 2.0
        hip_mid      = (LH_xy + RH_xy) / 2.0

        vx = RS_xy - LS_xy
        vy = hip_mid - shoulder_mid

        # ベクトルの正規化 (単位ベクトルにする)
        nx = np.linalg.norm(vx)
        ny = np.linalg.norm(vy)

        # normalize if valid
        if nx > 1e-6 and ny > 1e-6:
            vx = vx / nx
            vy = vy / ny
            axes_valid = True
            prev_vx, prev_vy = vx, vy

    else:
        # fallback: use previous axes to maintain continuity
        if prev_vx is not None:
            vx, vy = prev_vx, prev_vy
            axes_valid = True

    # ---------------------------------------------------------
    # Optical flow (Farnebäck)
    # 0.5: ピラミッドスケール（multi-scale 処理の縮小比）
    # 3: ピラミッドレベル数
    # 15: ウィンドウサイズ（周囲 15×15 ピクセルを参照）
    # 3: イテレーション数
    # 5: ポリノミアル展開近傍サイズ
    # 1.2 平滑化の係数
    # 0: オプションフラグ
    # ---------------------------------------------------------
    if flow_prev is not None:
        flow = cv2.calcOpticalFlowFarneback(
            flow_prev, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        # x方向とy方向の速度場
        fx = flow[...,0]
        fy = flow[...,1]

        # -----------------------------------------------------
        # Project optical flow into body axes
        # fx_body : movement along shoulder axis
        # fy_body : movement along torso axis
        # 画像の座標系 → 身体の座標系に変換される
        # 光フローの動きが 身体の左右方向・上下方向に分解される
        # -----------------------------------------------------
        if axes_valid:
            # 身体座標系の単位ベクトルを取り出す
            vx_x, vx_y = vx
            vy_x, vy_y = vy

            # 光フローの身体軸への投射
            # 光フローと肩軸ベクトルの内積
            fx_body = fx * vx_x + fy * vx_y
            fy_body = fx * vy_x + fy * vy_y

            # 身体軸での合成ベクトル
            mag_body = cv2.magnitude(fx_body, fy_body)

            # ROI内だけの平均値を計算
            vx_mean  = float(np.mean(fx_body[mask_bool]))
            vy_mean  = float(np.mean(fy_body[mask_bool]))
            mag_mean = float(np.mean(mag_body[mask_bool]))

            # -------------------------------------------------
            # Visualization (HSV optical flow)
            # arctan2でベクトルの向きを角度にする
            # 身体方向ベースの動きの向きを色で表現
            # H (色): 方向, S (彩度): 1-255, V (明度): 強さ(mag)
            # 右 (0度): 赤, 上 (90度): 緑, 左 (180度): 青, 下 (270度): 紫
            # -------------------------------------------------
            ang_body = np.arctan2(fy_body, fx_body)
            hsv = np.zeros((H,W,3), np.uint8)
            hsv[...,1] = 255
            
            # 色
            hsv[...,0] = ((ang_body + np.pi) / (2*np.pi) * 180).astype(np.uint8)
            
            # 明度 (動きの強さ), min-max正規化して0-255レンジに.
            hsv[...,2] = cv2.normalize(mag_body, None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)

            # HSVがらBGRに変換
            flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # ROIの部分だけ色を反映
            masked_flow = np.zeros_like(flow_color)
            masked_flow[mask_bool] = flow_color[mask_bool]

            # 元画像 35%, 光フロー可視化 65%
            overlay = cv2.addWeighted(frame, 0.35, masked_flow, 0.65, 0)

        else:
            vx_mean = vy_mean = mag_mean = np.nan
            overlay = frame.copy()

    else:
        # first frame (no flow yet)
        vx_mean = vy_mean = mag_mean = np.nan
        overlay = frame.copy()

    flow_prev = gray.copy()

    # ---------------------------------------------------------
    # MAD-based spike detection
    # アーチファクトの動きを除外
　　 # spike_flag: 0スパイクなし, 1スパイクあり
    # ---------------------------------------------------------
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

    # ROI outline (red = spike)
    color = (0,0,255) if (spike_flag and DRAW_SPIKE) else (0,255,255)
    cv2.polylines(overlay, [ROI_POINTS.astype(int)], True, color, 2)

    out.write(overlay)

    # Save metrics
    rows.append([frame_idx, t_sec, vx_mean, vy_mean, mag_mean, spike_flag])

    frame_idx += 1

# =============================================================
# Save results
# =============================================================
cap.release()
out.release()

df = pd.DataFrame(rows, columns=["frame","t_sec","vx_body","vy_body","mag_body","spike"])
df.to_csv(out_csv, index=False)

print("=== DONE ===")
print("Video:", out_video)
print("CSV  :", out_csv)
