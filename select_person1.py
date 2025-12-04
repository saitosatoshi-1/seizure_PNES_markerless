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
    """
    Select person1 at the best_bed frame and save PNG + JSON meta.

    Parameters
    ----------
    video_path : str
        Path to input silhouette video (e.g., QuickTime-safe MP4).
    gate_json_path : str
        Path to gate_config.json containing best_frame, bed_xyxy, bed_cxcy, gate_polygon, etc.
    out_png : str
        Path to save visualization PNG.
    out_json : str
        Path to save integrated meta JSON (bed + gate + person1).
    pose_model_path : str, optional
        YOLO pose model weight file path.
    conf : float, optional
        Confidence threshold for YOLO pose detection.
    """

    # 1) Load gate / bed configuration
    gate_json_path = str(gate_json_path)
    with open(gate_json_path, "r") as f:
        gate_cfg = json.load(f)

    best_frame_idx = int(gate_cfg["best_frame"])
    bx1, by1, bx2, by2 = map(float, gate_cfg["bed_xyxy"])
    bed_cx, bed_cy = map(float, gate_cfg["bed_cxcy"])
    gate_points = np.array(gate_cfg["gate_polygon"], dtype=np.float32)  # (N, 2)

    # 2) Read best_frame from video
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
    pose_model = YOLO(pose_model_path)
    res = pose_model.predict(frame, verbose=False, conf=conf)[0]

    persons = []

    # Preferred: use keypoints center
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
    candidates = [
        p for p in persons if inside_gate(p["cx"], p["cy"], gate_points)
    ]
    if len(candidates) == 0:
        # fallback: use all persons
        candidates = persons

    # 5) Among candidates, select nearest to bed center = person1
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

    # 7) Save PNG
    out_png = str(out_png)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(out_png, vis)
    print(
        f"[person1] frame={best_frame_idx} "
        f"bed_cxcy=({bed_cx:.1f},{bed_cy:.1f}) "
        f"bed_xyxy=({bx1:.1f},{by1:.1f},{bx2:.1f},{by2:.1f}) | "
        f"person1_idx={pidx} "
        f"person1_cxcy=({person1['cx']:.1f},{person1['cy']:.1f}) "
        f"dist_px={person1['dist']:.1f}  PNG:{ok}"
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


# -------------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Select person1 at best_bed frame using YOLO pose + gate_config.json"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="/content/FBTCS_silhouette_qt.mp4",
        help="Input silhouette video path",
    )
    parser.add_argument(
        "--gate-json",
        type=str,
        default="/content/gate_config.json",
        help="gate_config.json path",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default="/content/person1_at_bestbed.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="/content/bed_best.json",
        help="Output meta JSON path",
    )
    parser.add_argument(
        "--pose-model",
        type=str,
        default="yolo11l-pose.pt",
        help="YOLO pose model weights path",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="Confidence threshold for YOLO pose detection",
    )

    args = parser.parse_args()

    select_person1_at_best_bed(
        video_path=args.video,
        gate_json_path=args.gate_json,
        out_png=args.out_png,
        out_json=args.out_json,
        pose_model_path=args.pose_model,
        conf=args.conf,
    )


if __name__ == "__main__":
    main()
