import cv2, os

video_path = '/content/****.mp4'

#OpenCV(cv2)を使って動画フレーム単位で読み込む
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f'VideoCapture does not open: {video_path}')

orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


out_path = '/content/****out.mp4'

#一旦mp4vで書き出す
#OpenCVではmp4を直接h264で書くと壊れやすい→まずmp4vコーデックで保存するのが安定
#後でffmpegでQuickTime互換のH.264に変換する（Apple系OSでは重要）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter(out_path, fourcc, orig_fps, (width, height))
if not out.isOpened():
    raise RuntimeError('VideoWriter does not open')

#動画から1フレームずつ読み込む
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === シルエット推定を実行 ===
    results = seg_model(frame)

    # --- 可視化（人物マスクを塗りつぶした画像を返す） ---
    vis = results[0].plot()

    # --- 出力動画に書き込み ---
    #VideoWriter.write() は「1枚の画像を、動画の後ろに追加する」
    #while でフレームをfeedし続ける限り、動画が自動的にできる
    out.write(vis)

cap.release()
out.release()
print(f"YOLOセグメンテーション可視化の全体動画を保存しました: {out_try_path}")

# ===== QuickTime互換に再エンコード =====
out_qt_path = '/content/****_qt_seg.mp4'

#libx264 (標準的はH.264エンコーダ)
#yuv420p (互換性が高いピクセルフォーマット)
#baseline (他の機器でも再生可能)
#level 3.0 (低スペックデバイスでも再生可能)
#+faststart (即時再生が高速)
os.system(f"ffmpeg -y -i {out_try_path} -vcodec libx264 -pix_fmt yuv420p "
          f"-profile:v baseline -level 3.0 -movflags +faststart {out_qt_path}")
print("QuickTime互換ファイル:", out_qt_path)
