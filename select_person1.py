#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Select person1 at the best_bed frame using YOLO pose and gate_config.json.

Overview
--------
- Load gate_config.json (bed box, gate polygon, best_frame, etc.)
- Read one frame from the silhouette video at best_frame index
- Run YOLO pose detection on that frame
- For each detected person:
    - compute body center (mean of keypoints, or bbox center as fallback)
- Restrict candidates to inside the gate polygon (if any)
- Among candidates, select the one closest to the bed center → person1
- Visualize:
    - bed box / bed center
    - gate polygon
    - all persons (bbox & keypoints)
    - highlight person1 (red) and connect to bed center
- Save:
    - PNG image (for quick visual check)
    - JSON meta file (for downstream scripts)

Usage (example)
---------------
python select_person1_at_best_bed.py \\
    --video /content/FBTCS_silhouette_qt.mp4 \\
    --gate-json /content/gate_config.json \\
    --out-png /content/person1_at_bestbed.png \\
    --out-json /content/bed_best.json

Dependencies
------------
- Python 3.x
- numpy
- opencv-python
- ultralytics (YOLOv8/YOLO11)
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# -------------------------------------------------------------------------
# Geometry helpers
# -------------------------------------------------------------------------
def inside_gate(x: float, y: float, poly_2d: np.ndarray) -> bool:
    """
    Check whether the point (x, y) lies inside or on the boundary of the polygon.

    Parameters
    ----------
    x, y : float
        Coordinates of the point.
    poly_2d : (N, 2) np.ndarray
        Polygon vertices in image coordinates.

    Returns
    -------
    bool
    """
    if poly_2d is None or poly_2d.size == 0:
        return True  # if no polygon is defined, treat as always inside
    res = cv2.pointPolygonTest(
        poly_2d.reshape(-1, 1, 2),
        (float(x), float(y)),
        False
    )
    return res >= 0


# -------------------------------------------------------------------------
# Core logic
# -------------------------------------------------------------------------
def select_person1_at_best_bed(
    video_path: str,
    gate_json_path: str,
    out_png: str,
    out_json: str,
    pose_model_path: str = "yolo11l-pose.pt",
    conf: float = 0.2,
) -> None:

    # 1) Load gate / bed configuration
	# gate_cfg から以下を取り出す：
	# best_frame →「ベッド位置が一番きれいに見えるフレーム番号」
	# bed_xyxy → ベッド矩形（x1, y1, x2, y2）
	# bed_cxcy → ベッド中心座標
	# gate_polygon → ベッド周囲の Gate ポリゴン（患者がいてほしい領域）
    gate_json_path = str(gate_json_path)
    with open(gate_json_path, "r") as f:
        gate_cfg = json.load(f)

    best_frame_idx = int(gate_cfg["best_frame"])
    bx1, by1, bx2, by2 = map(float, gate_cfg["bed_xyxy"])
    bed_cx, bed_cy = map(float, gate_cfg["bed_cxcy"])
    gate_points = np.array(gate_cfg["gate_polygon"], dtype=np.float32)  # (N, 2)

    
    # 2) Read best_frame from video
    # 動画を VideoCapture で開く
	# best_frame_idx にシークして、そのフレームを1枚読む
	# 読めなければエラー
	# フレームのサイズ H, W を取得 → 「ベッドが一番きれいに写っているフレーム」を1枚だけ取り出す。
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Failed to read best_frame_idx={best_frame_idx}")

    H, W = frame.shape[:2]

    
    # 3) YOLO pose detection
    # YOLO pose モデルをロード
	# frame に対して推論
	# res が、このフレーム内の keypoints / bbox 情報を持つ結果オブジェクト
    pose_model = YOLO(pose_model_path)
    res = pose_model.predict(frame, verbose=False, conf=conf)[0]

    persons = []


    # Preferred: use keypoints center
    # res.keypoints.xy : (人数 N, キーポイント数 K, 2) の配列
	# 各人物 i について：
	# NaN を除外した有効な keypoints だけを使う
	# その平均位置 (cx, cy) を取る
	# {"idx": i, "cx": ..., "cy": ...} として persons に追加 → 「人物 i の重心（おおざっぱな体の中心）」 を計算している。

    if res.keypoints is not None and res.keypoints.xy is not None:
        kps = res.keypoints.xy.cpu().numpy()  # shape: (N, K, 2)
        for i in range(kps.shape[0]):
            kp = kps[i]
            valid = ~np.isnan(kp).any(axis=1)
            if not np.any(valid):
                continue
            cx, cy = np.nanmean(kp[valid], axis=0)
            persons.append({"idx": i, "cx": float(cx), "cy": float(cy)})

    # Fallback: bbox center if no keypoints were available
    if not persons and res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = xyxy[i]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            persons.append({"idx": i, "cx": float(cx), "cy": float(cy)})

    if len(persons) == 0:
        raise RuntimeError("No person detected in the best_frame.")

    # 4) Restrict to inside gate polygon if possible
    # inside_gate(x, y, gate_points)（別関数）を使って、
	# 体中心が Gate ポリゴンの中にある人物だけを候補にする
	# （Gate内=ベッドの上 or その周囲→患者らしい場所）
	# Gate 内に誰もいなければ、仕方ないので全員を候補に使う

    candidates = [
        p for p in persons if inside_gate(p["cx"], p["cy"], gate_points)
    ]
    if len(candidates) == 0:
        # fallback: use all persons
        candidates = persons


    # 5) Among candidates, select nearest to bed center = person1
    # 各候補 p について：
	# ベッド中心 (bed_cx, bed_cy) との距離 dist を計算
	# ベッド中心に一番近い人物 を person1 とする
	# その idx（YOLO 内の人物インデックス）を pidx として保存 → **「Gate 内でベッドに一番近い人 = ベッドの患者本人」**というルール。

    for p in candidates:
        dx = p["cx"] - bed_cx
        dy = p["cy"] - bed_cy
        p["dist"] = math.hypot(dx, dy)

    person1 = min(candidates, key=lambda d: d["dist"])
    pidx = int(person1["idx"])


    # 6) Visualization
    vis = frame.copy()

    # bed box (green)
    cv2.rectangle(
        vis,
        (int(round(bx1)), int(round(by1))),
        (int(round(bx2)), int(round(by2))),
        (0, 255, 0),
        3,
    )
    cv2.circle(
        vis,
        (int(round(bed_cx)), int(round(bed_cy))),
        6,
        (0, 255, 0),
        -1,
    )
    cv2.putText(
        vis,
        "best bed",
        (int(round(bx1)), max(0, int(round(by1)) - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # gate polygon (dark green)
    if gate_points.size > 0:
        cv2.polylines(
            vis,
            [gate_points.astype(np.int32)],
            True,
            (0, 180, 0),
            2,
        )

    # all bboxes (gray)
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = map(int, xyxy[i])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (200, 200, 200), 2)

    # all keypoints (gray)
    if res.keypoints is not None and res.keypoints.xy is not None:
        kps = res.keypoints.xy.cpu().numpy()
        for i in range(kps.shape[0]):
            for (x, y) in kps[i]:
                if not (np.isnan(x) or np.isnan(y)):
                    cv2.circle(
                        vis,
                        (int(x), int(y)),
                        2,
                        (180, 180, 180),
                        -1,
                    )

    # highlight person1 (red)
    if res.boxes is not None and len(res.boxes) > pidx:
        x1, y1, x2, y2 = map(int, res.boxes.xyxy.cpu().numpy()[pidx])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)

    cv2.circle(
        vis,
        (int(round(person1["cx"])), int(round(person1["cy"]))),
        6,
        (0, 0, 255),
        -1,
    )
    cv2.line(
        vis,
        (int(round(bed_cx)), int(round(bed_cy))),
        (int(round(person1["cx"])), int(round(person1["cy"]))),
        (0, 0, 255),
        2,
    )
    cv2.putText(
        vis,
        f"person1 idx={pidx} dist={person1['dist']:.1f}px",
        (int(round(person1["cx"])) + 8, int(round(person1["cy"]))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    # 8) Save integrated meta JSON (bed + gate + person1)
    meta = {
        "best_frame": best_frame_idx,
        "bed_xyxy": [float(bx1), float(by1), float(bx2), float(by2)],
        "bed_cxcy": [float(bed_cx), float(bed_cy)],
        "gate_polygon": gate_cfg.get("gate_polygon", []),
        "gate_margins": gate_cfg.get("margins", []),
        "person1_idx": pidx,
        "person1_cx": float(person1["cx"]),
        "person1_cy": float(person1["cy"]),
        "person1_dist_px": float(person1["dist"]),
        "frame_width": int(W),
        "frame_height": int(H),
        "video_path": str(video_path),
        "gate_json": str(gate_json_path),
        "pose_model": str(pose_model_path),
    }

    out_json = str(out_json)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("[Info] Saved meta JSON:", out_json)


