#シルエット推定


import cv2, os

# ===== 入力動画（QuickTime互換版を使用） =====
video_path = '/content/****qt.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f'VideoCaptureが開けません: {video_path}')

orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



# ===== 出力設定 =====
out_try_path = '/content/******.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 一旦 mp4v で書き出し
out = cv2.VideoWriter(out_try_path, fourcc, orig_fps, (width, height))
if not out.isOpened():
    raise RuntimeError('VideoWriterが開けません')

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === シルエット推定を実行 ===
    results = seg_model(frame)

    # --- 可視化（人物マスクを塗りつぶした画像を返す） ---
    vis = results[0].plot()

    # --- 出力動画に書き込み ---
    out.write(vis)

    frame_idx += 1

cap.release()
out.release()
print(f"YOLOセグメンテーション可視化の全体動画を保存しました: {out_try_path}")



# ===== QuickTime互換に再エンコード =====
out_qt_path = '/content/FBTCS_silhouette_qt_seg.mp4'
os.system(f"ffmpeg -y -i {out_try_path} -vcodec libx264 -pix_fmt yuv420p "
          f"-profile:v baseline -level 3.0 -movflags +faststart {out_qt_path}")
print("QuickTime互換ファイル:", out_qt_path)
