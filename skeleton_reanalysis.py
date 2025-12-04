# ==========================================================
# Skeleton + PC1 amplitude overlay movie
# ==========================================================

import numpy as np
import cv2
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io

# ----------------------------------------------------------
# 入力ファイル
# ----------------------------------------------------------
video_path = "/content/FBTCS_silhouette_qt.mp4"
refined_npz_path = "/content/person1_refined_kpts.npz"

# 出力ファイル
out_skel_pc1_path = "/content/person1_skeleton_pc1_overlay.mp4"

# ----------------------------------------------------------
# refined kpts 読み込み
# ----------------------------------------------------------
dat = np.load(refined_npz_path)
LS = dat["LS_refined"]
RS = dat["RS_refined"]
LH = dat["LH_refined"]
RH = dat["RH_refined"]
t_sec = dat["t_sec"]

# PC1 amplitude の計算（あなたの解析から取得したい場合は置き換えてOK）
# 今回は例として shoulder-midpoint のY変位をPC1代わりに使用
shoulder_mid = (LS + RS) / 2
hip_mid = (LH + RH) / 2

# 簡易PC1（肩と股関節を使った主成分風）
pc1 = np.linalg.norm(shoulder_mid - hip_mid, axis=1)

# 正規化
pc1 = pc1 - np.nanmean(pc1)

# ----------------------------------------------------------
# 動画読み込み
# ----------------------------------------------------------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 描画サイズ
plot_w = 500
plot_h = H

out = cv2.VideoWriter(
    out_skel_pc1_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (W + plot_w, H)
)

# ----------------------------------------------------------
# Helper: skeleton 描画
# ----------------------------------------------------------
def draw_skeleton(frame, LS, RS, LH, RH):
    f = frame.copy()

    # keypoint valid check
    def draw_point(p, color):
        if np.all(np.isfinite(p)):
            cv2.circle(f, (int(p[0]), int(p[1])), 6, color, -1, cv2.LINE_AA)

    def draw_line(p1, p2, color):
        if np.all(np.isfinite(p1)) and np.all(np.isfinite(p2)):
            cv2.line(f, (int(p1[0]), int(p1[1])),
                        (int(p2[0]), int(p2[1])),
                        color, 3, cv2.LINE_AA)

    # upper body
    draw_point(LS, (0,255,0))
    draw_point(RS, (0,255,0))
    draw_line(LS, RS, (0,200,0))

    # lower body
    draw_point(LH, (255,0,0))
    draw_point(RH, (255,0,0))
    draw_line(LH, RH, (200,0,0))

    # torso line
    mid_s = (LS+RS)/2
    mid_h = (LH+RH)/2
    draw_line(mid_s, mid_h, (255,255,0))

    return f

# ----------------------------------------------------------
# Main loop: 動画 + PC1 波形の合成
# ----------------------------------------------------------
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx >= len(pc1):
        break

    # ---- 左：Skeleton overlay ----
    f_skel = draw_skeleton(frame, LS[frame_idx], RS[frame_idx],
                                   LH[frame_idx], RH[frame_idx])

    # ---- 右：PC1 plot ----
    fig = Figure(figsize=(4, 3), dpi=150)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    # 軸ラベル非表示
    ax.set_xticks([])
    ax.set_yticks([])

    ax.plot(t_sec, pc1, color="black", linewidth=1)
    ax.axvline(t_sec[frame_idx], color="red", linewidth=2)  # 現在時刻を強調

    fig.tight_layout()

    canvas.draw()
    plot_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    plot_img = plot_img.reshape(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), 3)

    # Matplotlib の画像を OpenCV サイズに合わせる
    plot_img = cv2.resize(plot_img, (plot_w, H))

    # ---- 左右結合 ----
    combined = np.hstack([f_skel, plot_img])

    out.write(combined)
    frame_idx += 1

cap.release()
out.release()

print("=== DONE ===")
print("[Skeleton + PC1 video]", out_skel_pc1_path)
