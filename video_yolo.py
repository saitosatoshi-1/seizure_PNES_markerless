import cv2, os, numpy as np
from ultralytics import YOLO

# ====== 入力動画 ======
video_path = '/content/*****.mp4'
if not os.path.exists(video_path):
    raise FileNotFoundError(f'動画が見つかりません: {video_path}')

# ====== モデル ======
det_model = YOLO('yolo11s.pt')
seg_model = YOLO('yolo11s-seg.pt')

# ====== 動画オープン ======
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError('VideoCaptureが開けません')

ret, test_frame = cap.read()
if not ret:
    raise RuntimeError('最初のフレームを読み込めません')
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
height, width = test_frame.shape[:2]

print(f"fps={orig_fps:.3f}, frames={frame_count}, size={width}x{height}")

# ====== 出力設定 ======
target_fps = min(20.0, orig_fps)
frame_interval = max(1, int(round(orig_fps / target_fps)))
out_fps = max(1, int(round(orig_fps / frame_interval)))

out_path_tmp = '/content/FBTCS_tmp.mp4'   # OpenCV出力（互換性低い）
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4vならまず書ける
out = cv2.VideoWriter(out_path_tmp, fourcc, out_fps, (width, height))

if not out.isOpened():
    raise RuntimeError('VideoWriterが開けません')



# ====== ダミー処理ループ（間引きしながらコピー） ======
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

print("OpenCVで一旦 mp4v 出力完了:", out_path_tmp)




# ====== ffmpegでQuickTime互換に再エンコード ======
out_path_qt = '/content/FBTCS_silhouette_qt.mp4'
os.system(f"ffmpeg -y -i {out_path_tmp} -vcodec libx264 -pix_fmt yuv420p "
          f"-profile:v baseline -level 3.0 -movflags +faststart {out_path_qt}")

print("QuickTime互換ファイル出力:", out_path_qt)
