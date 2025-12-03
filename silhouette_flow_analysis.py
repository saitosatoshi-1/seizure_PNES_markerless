# =========================
# Skeleton + Silhouette Flow（最適化版）
#  - gate_config.json: priority / allow / exclude のゾーン適用 + ゾーン・ヒステリシス
#  - SkeletonベースROI（体幹カプセル）で候補を強制フィルタ → 器物/他人への乗り移り抑制
#  - 冒頭はアンカー厳格（priority近傍 + ROI一致）
#  - 時間同期: CAP_PROP_POS_MSEC → np.searchsorted（ANCHOR_SHIFT_SECで微調整）
#  - 出力: 動画オーバーレイ + CSV（raw/clean）
# =========================

import os, json
import numpy as np
import cv2
from ultralytics import YOLO
import pandas as pd

# ===== 入出力 =====
video_path   = '/content/FBTCS_silhouette_qt.mp4'
npz_path     = '/content/person1_clean_kpts.npz'
gate_json    = '/content/gate_config.json'
csv_path     = '/content/person1_silhouette_flow.csv'
stats_csv    = '/content/person1_flow_stats.csv'
out_try_path = '/content/person1_silhouette_seg_clean.mp4'
out_qt_path  = '/content/person1_silhouette_seg_clean_qt.mp4'

# ===== 可視化 =====
COLOR_UPPER_FILL=(60,200,60);   COLOR_LOWER_FILL=(200,60,200)
COLOR_SPLIT=(0,255,255);        COLOR_SEL=(80,255,80); COLOR_CNT=(170,170,170)
COLOR_ZONE_PRIO=(0,200,255);    COLOR_ZONE_ALLOW=(120,120,255); COLOR_ZONE_EXCL=(60,60,60)
ALPHA_UP=0.40; ALPHA_LOW=0.40;  ALPHA_HEAT=0.30; HEAT_PCTL=95
DRAW_ALL_CONTOURS=True; SHOW_HEAT=True; SHOW_ZONES=True

# ===== パラメータ =====
CONF_MIN=0.25; KERNEL_MORPH=(3,3)
T0_SEC=0.0; MISS_MAX_SEC=1.0; MEDIAN_WIN_FR=2; PCLIP_LOW=1; PCLIP_HIGH=99
USE_POS_MSEC=True; ANCHOR_SHIFT_SEC=0.0

# アンカー厳格ウォームアップ
LOCK_WARMUP_SEC=3.0
ANCHOR_RADIUS_RATIO=0.10
ANCHOR_CENT_MAX_RATIO=0.12

# 乗り移り抑制（+ゾーン補強）
IOU_KEEP=0.35; DIST_KEEP_RATIO=0.20
IOU_SWITCH=0.15; DIST_SWITCH_RATIO=0.35
STICKY_FRAMES=8
ZONE_STICKY_FRAMES=12
MIN_AREA_RATIO=0.005; MIN_HEIGHT_RATIO=0.15; MAX_ASPECT=6.0

# Skeleton ROI（体幹カプセル）関連
POSE_OVERLAP_MIN = 0.55   # 候補はROI被覆率がこの値以上を推奨
POSE_BONUS       = 2.0    # ROI一致の加点
POSE_STRICT_WARMUP = True # ウォームアップ中はROI未一致を即棄却
CAPSULE_RADIUS_SCALE = 0.6 # 体幹カプセル半径 = 肩幅/股幅中央値 × 係数

# ===== モデル =====
seg_model = YOLO('yolo11s-seg.pt')

# ===== 動画IO =====
cap = cv2.VideoCapture(video_path)
if not cap.isOpened(): raise RuntimeError(f'open fail: {video_path}')
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(out_try_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W,H))
YY, XX = np.mgrid[0:H, 0:W]
k_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, KERNEL_MORPH)

# ===== npz 読み込み（必要ならスケール） =====
dat = np.load(npz_path)
time_all = dat["time_all"]; LS, RS = dat["LS"], dat["RS"]; LH, RH = dat["LH"], dat["RH"]
refW=dat["frame_w"].item() if "frame_w" in dat.files else (dat["ref_w"].item() if "ref_w" in dat.files else W)
refH=dat["frame_h"].item() if "frame_h" in dat.files else (dat["ref_h"].item() if "ref_h" in dat.files else H)
if (refW,refH)!=(W,H):
    sx, sy = W/refW, H/refH
    for A in (LS,RS,LH,RH):
        A[:,0]*=sx; A[:,1]*=sy
hip_mid=(LH+RH)/2.0; shoulder_mid=(LS+RS)/2.0
shoulder_w_med=float(np.nanmedian(np.linalg.norm(RS-LS,axis=1)))
hip_w_med=float(np.nanmedian(np.linalg.norm(RH-LH,axis=1)))

# ===== gate_config.json 読み込み =====
def _as_polys(obj):
    # [[x,y],...] or [[[x,y],...], ...] を許容
    if obj is None: return []
    if isinstance(obj, dict):
        for k in ("priority","allow","exclude","bed_poly","polygon","gate","poly","points"):
            if k in obj and isinstance(obj[k], (list,tuple)): obj = obj[k]; break
    if isinstance(obj, list) and obj and isinstance(obj[0], (list,tuple)):
        if obj and isinstance(obj[0][0], (list,tuple)):  # 複数ポリゴン
            return [np.array(p,dtype=float) for p in obj]
        return [np.array(obj,dtype=float)]
    return []

def load_gate_masks(json_path, W, H, refW=None, refH=None):
    if not os.path.isfile(json_path):
        return None, None, None
    with open(json_path,'r') as f: js=json.load(f)

    # JSON側の参照解像度
    jrefW = js.get("ref_w", js.get("frame_w", refW))
    jrefH = js.get("ref_h", js.get("frame_h", refH))
    if jrefW and jrefH:
        refW, refH = jrefW, jrefH

    def scale_poly(P):
        if P is None: return None
        Q = P.copy()
        if (refW and refH) and ((refW,refH)!=(W,H)):
            Q[:,0]*=W/float(refW); Q[:,1]*=H/float(refH)
        return Q

    pri_list = _as_polys(js.get("priority")) or _as_polys(js.get("bed_poly")) or _as_polys(js.get("polygon"))
    allow_list = _as_polys(js.get("allow"))
    excl_list  = _as_polys(js.get("exclude"))

    # allow 未指定なら全体
    allow_mask = np.ones((H,W), np.uint8) if not allow_list else np.zeros((H,W), np.uint8)
    for P in allow_list or []:
        P=scale_poly(P); cv2.fillPoly(allow_mask, [P.astype(np.int32)], 1)

    pri_mask = np.zeros((H,W), np.uint8)
    for P in pri_list or []:
        P=scale_poly(P); cv2.fillPoly(pri_mask, [P.astype(np.int32)], 1)

    exc_mask = np.zeros((H,W), np.uint8)
    for P in excl_list or []:
        P=scale_poly(P); cv2.fillPoly(exc_mask, [P.astype(np.int32)], 1)

    return pri_mask.astype(bool), allow_mask.astype(bool), exc_mask.astype(bool)

priority_mask, allow_mask, exclude_mask = load_gate_masks(gate_json, W, H, refW, refH)

# ===== ユーティリティ =====
def mcent(m):
    ys,xs=np.nonzero(m);
    return None if xs.size==0 else (float(xs.mean()), float(ys.mean()))

def draw_cnt(img, m, col, th=2):
    if m is None or not m.any(): return
    m8=(m.astype(np.uint8)*255); cnts,_=cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts: cv2.drawContours(img,cnts,-1,col,th)

def draw_poly_mask(img, mask, color, thickness=2):
    if mask is None: return
    cnts,_=cv2.findContours((mask.astype(np.uint8)*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts: cv2.drawContours(img, cnts, -1, color, thickness)

def overlay(img, m, col, a):
    if m is None or not m.any(): return
    ov=img.copy(); ov[m]=(ov[m]*(1-a)+np.array(col)*a).astype(np.uint8); img[:]=ov

def heat_overlay(img, mag, m, a, pctl):
    if m is None or not m.any(): return
    vmax=np.nanpercentile(mag[m],pctl) if np.isfinite(mag[m]).any() else 1.0; vmax=max(vmax,1e-6)
    heat=(np.clip(mag/vmax,0,1)*255).astype(np.uint8); cmap=cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    ov=img.copy(); ov[m]=(cmap[m]*a+img[m]*(1-a)).astype(np.uint8); img[:]=ov

def iou(a,b):
    if a is None or b is None: return 0.0
    inter=np.logical_and(a,b).sum(); uni=np.logical_or(a,b).sum()
    return inter/(uni+1e-6)

def clip_to_allow(m):
    return np.logical_and(m, allow_mask) if allow_mask is not None else m

def is_ok_shape(m):
    area=int(m.sum())
    if area < MIN_AREA_RATIO*(W*H): return False
    ys,xs=np.nonzero(m); x1,x2=xs.min(),xs.max(); y1,y2=ys.min(),ys.max()
    w=max(1,x2-x1+1); h=max(1,y2-y1+1)
    if max(w/h,h/w) > MAX_ASPECT: return False
    if h < MIN_HEIGHT_RATIO*H: return False
    return True

def zone_of_mask(m):
    if m is None: return "none"
    if priority_mask is not None and (m & priority_mask).any(): return "priority"
    if allow_mask is not None and (m & allow_mask).any(): return "allow"
    return "none"

def candidate_zone_bonus(m):
    if exclude_mask is not None and (m & exclude_mask).any():
        return None
    if priority_mask is not None and (m & priority_mask).any():
        return 1.0
    if allow_mask is not None and (m & allow_mask).any():
        return 0.2
    return None

# ==== Skeletonから体幹カプセルROIを作成 ====
def capsule_mask_from_pose(W, H, shoulder_xy, hip_xy,
                           shoulder_w_med, hip_w_med, scale_rad=CAPSULE_RADIUS_SCALE):
    canvas = np.zeros((H, W), np.uint8)
    p1 = np.asarray(shoulder_xy, float); p2 = np.asarray(hip_xy, float)
    if not (np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))): return canvas.astype(bool)
    r1 = max(6.0, float(shoulder_w_med) * scale_rad)
    r2 = max(6.0, float(hip_w_med) * scale_rad)
    # 円＋太線でカプセル近似
    cv2.circle(canvas, (int(round(p1[0])), int(round(p1[1]))), int(round(r1)), 255, -1)
    cv2.circle(canvas, (int(round(p2[0])), int(round(p2[1]))), int(round(r2)), 255, -1)
    thickness = max(1, int(round(r1 + r2)))
    cv2.line(canvas, (int(round(p1[0])), int(round(p1[1]))),
             (int(round(p2[0])), int(round(p2[1]))), 255, thickness)
    # ゾーンで切り出しておく（allow外はゼロ）
    M = (canvas>0)
    return np.logical_and(M, allow_mask) if allow_mask is not None else M

def overlap_ratio(mask_roi, mask_cand):
    if mask_roi is None or mask_cand is None: return 0.0
    inter = np.logical_and(mask_roi, mask_cand).sum()
    denom = mask_cand.sum() + 1e-6
    return float(inter) / float(denom)

# ===== 選択（IoU + 距離 + アンカー + ROI + ゾーン粘り） =====
def select_mask(prev_m, prev_zone, cand_list, t_sec, anchor_xy, sticky, zone_sticky, pose_roi):
    diag=np.hypot(W,H)
    warmup = (t_sec <= LOCK_WARMUP_SEC) and (anchor_xy is not None)
    best=None; best_score=-1.0
    prev_c = mcent(prev_m) if prev_m is not None else None
    r = ANCHOR_RADIUS_RATIO*diag if warmup else None

    for m in cand_list:
        m = clip_to_allow(m)
        if exclude_mask is not None and (m & exclude_mask).any():
            continue
        if not is_ok_shape(m):
            continue

        # ★ Skeleton ROI との一致で強制フィルタ
        pose_ov = overlap_ratio(pose_roi, m) if pose_roi is not None else 0.0
        if pose_roi is not None:
            if pose_ov < POSE_OVERLAP_MIN and (POSE_STRICT_WARMUP or warmup):
                continue  # ウォームアップ中はより厳格

        c = mcent(m)
        if c is None:
            continue

        # ウォームアップ：priority 内 + アンカー近傍
        if warmup:
            if priority_mask is not None and not (m & priority_mask).any(): continue
            if np.linalg.norm(np.array(c)-np.array(anchor_xy)) > r: continue
            if (np.linalg.norm(np.array(c)-np.array(anchor_xy))/(diag+1e-6)) > ANCHOR_CENT_MAX_RATIO: continue

        # 基本スコア
        j = iou(prev_m, m) if prev_m is not None else 0.0
        d_ratio = 1.0 if prev_c is None else np.linalg.norm(np.array(c)-np.array(prev_c))/(diag+1e-6)
        score = 2.0*j + (1.0 - min(1.0, d_ratio))

        # アンカー近さ
        if anchor_xy is not None:
            da = np.linalg.norm(np.array(c)-np.array(anchor_xy))/(diag+1e-6)
            score += 0.8*(1.0 - min(1.0, da))

        # ゾーンボーナス
        z_bonus = candidate_zone_bonus(m)
        if z_bonus is None:
            continue
        score += z_bonus

        # ★ ROI一致の加点（重なりに応じて）
        if pose_roi is not None:
            score += POSE_BONUS * pose_ov
            if pose_ov < POSE_OVERLAP_MIN:
                score -= 0.5   # 軽い減点

        # ゾーン・ヒステリシス
        cand_zone = zone_of_mask(m)
        if prev_zone == "priority" and cand_zone == "priority": score += 0.5
        elif prev_zone == "priority" and cand_zone != "priority": score -= 0.3

        if score>best_score:
            best_score=score; best=(m, j, d_ratio, cand_zone)

    if best is None:
        return prev_m, prev_zone, sticky, zone_sticky, True

    m, j, d_ratio, cand_zone = best
    keep_like   = (j>=IOU_KEEP) or (d_ratio<=DIST_KEEP_RATIO)
    switch_hard = (j<=IOU_SWITCH) and (d_ratio>=DIST_SWITCH_RATIO)

    if keep_like:
        sticky = min(STICKY_FRAMES, sticky+1)
    elif switch_hard and not warmup:
        sticky = 0
    else:
        if sticky < STICKY_FRAMES:
            return prev_m if prev_m is not None else m, prev_zone if prev_zone else cand_zone, sticky+1, zone_sticky, True
        sticky = 0

    # ゾーン・ヒステリシス（priority離脱は粘る）
    if prev_zone == "priority" and cand_zone != "priority" and not warmup:
        if zone_sticky < ZONE_STICKY_FRAMES and not switch_hard:
            return prev_m if prev_m is not None else m, prev_zone, sticky, zone_sticky+1, True
        zone_sticky = 0
    else:
        zone_sticky = min(ZONE_STICKY_FRAMES, zone_sticky+1) if cand_zone == "priority" else zone_sticky

    return m, cand_zone, sticky, zone_sticky, False

# ===== 後処理ユーティリティ =====
def series_clip_interp_medfilt(x, fps_video, p1=1, p99=99, max_gap_sec=1.0, median_win_fr=2):
    x=np.asarray(x,float); out=x.copy(); n=len(out)
    m=np.isfinite(out)
    if m.sum()>=10:
        lo,hi=np.nanpercentile(out[m],[p1,p99]); out[(out<lo)|(out>hi)]=np.nan
    max_gap=max(1,int(round(fps_video*max_gap_sec)))
    i=0; n_interp=0
    while i<n:
        if not np.isfinite(out[i]):
            j=i
            while j<n and not np.isfinite(out[j]): j+=1
            gap=j-i
            if gap<=max_gap and i>0 and j<n and np.isfinite(out[i-1]) and np.isfinite(out[j]):
                out[i:j]=np.linspace(out[i-1],out[j],gap+2)[1:-1]; n_interp+=gap
            i=j
        else: i+=1
    if median_win_fr>=2 and n>=3:
        med=out.copy(); med[1:-1]=np.nanmedian(np.vstack([out[:-2],out[1:-1],out[2:]]),axis=0); out=med
    n_nan_after=np.count_nonzero(~np.isfinite(out))
    return out, dict(total=n, nan_before=np.count_nonzero(~np.isfinite(x)),
                     nan_after=n_nan_after, n_interp=n_interp, excl_rate=n_nan_after/max(n,1))

# ===== メイン =====
frames=[]; t_secs=[]; flow_prev_gray=None
flow_all_raw=[]; flow_upper_raw=[]; flow_lower_raw=[]
all_vx_raw=[]; all_vy_raw=[]; upper_vx_raw=[]; upper_vy_raw=[]; lower_vx_raw=[]; lower_vy_raw=[]
prev_mask=None; prev_zone="none"; sticky=0; zone_sticky=0
miss_run=0; MISS_MAX=int(round(MISS_MAX_SEC*fps))

frame_idx=0
while True:
    ret, frame = cap.read()
    if not ret: break

    # 実時間（VFR対策）
    if USE_POS_MSEC:
        tm = cap.get(cv2.CAP_PROP_POS_MSEC)
        t_sec = (tm/1000.0) if (tm and tm>0) else (frame_idx/fps)
    else:
        t_sec = frame_idx/fps

    # Skeleton 時間同期（+シフト）
    t_anchor = t_sec + ANCHOR_SHIFT_SEC
    ai = int(np.clip(np.searchsorted(time_all, t_anchor, side='right')-1, 0, len(time_all)-1))

    # 体幹カプセルROI（肩中点→股中点）
    pose_roi = None
    if np.all(np.isfinite(shoulder_mid[ai])) and np.all(np.isfinite(hip_mid[ai])):
        pose_roi = capsule_mask_from_pose(
            W, H,
            (float(shoulder_mid[ai,0]), float(shoulder_mid[ai,1])),
            (float(hip_mid[ai,0]),      float(hip_mid[ai,1])),
            shoulder_w_med, hip_w_med, CAPSULE_RADIUS_SCALE
        )

    # アンカー（股中点）
    anchor_xy = (float(hip_mid[ai,0]), float(hip_mid[ai,1])) if np.all(np.isfinite(hip_mid[ai])) else None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # セグ推論
    res = seg_model.predict(frame, verbose=False, conf=CONF_MIN)[0]
    masks_raw = res.masks.data.cpu().numpy() if (hasattr(res,'masks') and res.masks is not None) else None
    if masks_raw is not None and masks_raw.shape[1:]!=(H,W):
        masks = np.stack([cv2.resize(m.astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool) for m in masks_raw])
    else:
        masks = masks_raw.astype(bool) if masks_raw is not None else None

    # ゾーン図示
    if SHOW_ZONES:
        draw_poly_mask(frame, priority_mask, COLOR_ZONE_PRIO, 2)
        draw_poly_mask(frame, allow_mask,    COLOR_ZONE_ALLOW, 2)
        draw_poly_mask(frame, exclude_mask,  COLOR_ZONE_EXCL,  2)

    if DRAW_ALL_CONTOURS and (masks is not None):
        for i in range(masks.shape[0]): draw_cnt(frame, masks[i], COLOR_CNT, 1)

    # 候補整形（open→close）
    cand=[]
    if masks is not None:
        for i in range(masks.shape[0]):
            m = (masks[i].astype(np.uint8)*255)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_morph, iterations=1)>0
            m = cv2.morphologyEx(m.astype(np.uint8)*255, cv2.MORPH_CLOSE, k_morph, iterations=1)>0
            cand.append(m.astype(bool))

    # 1人選択（ROI & ゾーン粘り込み）
    sel_all=None; as_missing=True
    if cand:
        sel_all, prev_zone, sticky, zone_sticky, as_missing = select_mask(
            prev_mask, prev_zone, cand, t_sec, anchor_xy, sticky, zone_sticky, pose_roi
        )
        if sel_all is not None: draw_cnt(frame, sel_all, COLOR_SEL, 2)

    miss_run = miss_run+1 if as_missing else 0

    # 上下分割（股スパン直交, 時間同期）
    sel_upper=sel_lower=None
    if sel_all is not None and np.all(np.isfinite(LH[ai])) and np.all(np.isfinite(RH[ai])):
        lh, rh = LH[ai], RH[ai]
        mid=(lh+rh)/2.0; vec=rh-lh; nrm=np.linalg.norm(vec)
        if nrm>1e-6:
            perp=np.array([-vec[1], vec[0]],float)/nrm
            side=(XX-mid[0])*perp[0]+(YY-mid[1])*perp[1]
            sel_upper = sel_all & (side<=0); sel_lower = sel_all & (side>0)
            v=vec/nrm; p1=(int(mid[0]-v[0]*1000),int(mid[1]-v[1]*1000)); p2=(int(mid[0]+v[0]*1000),int(mid[1]+v[1]*1000))
            cv2.line(frame,p1,p2,COLOR_SPLIT,2); cv2.circle(frame,(int(mid[0]),int(mid[1])),4,COLOR_SPLIT,-1)

    if sel_upper is not None:
        overlay(frame, sel_upper, COLOR_UPPER_FILL, ALPHA_UP); draw_cnt(frame, sel_upper, (255,255,255), 2)
    if sel_lower is not None:
        overlay(frame, sel_lower, COLOR_LOWER_FILL, ALPHA_LOW); draw_cnt(frame, sel_lower, (255,255,255), 2)

    # 光フロー
    fa=fu=fl=np.nan; ux_all=uy_all=ux_u=uy_u=ux_l=uy_l=np.nan
    if flow_prev_gray is not None:
        flow=cv2.calcOpticalFlowFarneback(flow_prev_gray, gray, None, 0.5,3,15,3,5,1.2,0)
        fx,fy = flow[...,0], flow[...,1]; mag=cv2.magnitude(fx,fy)
        if SHOW_HEAT and (sel_all is not None): heat_overlay(frame, mag, sel_all, ALPHA_HEAT, HEAT_PCTL)

        if sel_all is not None and sel_all.any():  fa=float(np.nanmean(mag[sel_all])); ux_all=float(np.nanmean(fx[sel_all])); uy_all=float(np.nanmean(fy[sel_all]))
        if sel_upper is not None and sel_upper.any(): fu=float(np.nanmean(mag[sel_upper])); ux_u=float(np.nanmean(fx[sel_upper])); uy_u=float(np.nanmean(fy[sel_upper]))
        if sel_lower is not None and sel_lower.any(): fl=float(np.nanmean(mag[sel_lower])); ux_l=float(np.nanmean(fx[sel_lower])); uy_l=float(np.nanmean(fy[sel_lower]))

        # 正規化 + Y上向き正
        fa/=shoulder_w_med; fu/=shoulder_w_med; fl/=hip_w_med
        ux_all/=shoulder_w_med; uy_all=-uy_all/shoulder_w_med
        ux_u  /=shoulder_w_med; uy_u  =-uy_u  /shoulder_w_med
        ux_l  /=hip_w_med;      uy_l  =-uy_l  /hip_w_med

    flow_prev_gray=gray.copy()
    use_flag=int((t_sec>=T0_SEC) and (sel_all is not None) and (miss_run < int(round(MISS_MAX_SEC*fps))))

    frames.append(frame_idx); t_secs.append(t_sec)
    flow_all_raw.append(fa); flow_upper_raw.append(fu); flow_lower_raw.append(fl)
    all_vx_raw.append(ux_all); all_vy_raw.append(uy_all)
    upper_vx_raw.append(ux_u); upper_vy_raw.append(uy_u)
    lower_vx_raw.append(ux_l); lower_vy_raw.append(uy_l)

    out.write(frame); frame_idx+=1; prev_mask = sel_all if sel_all is not None else prev_mask

cap.release(); out.release()

# ===== 後処理(clean) =====
df=pd.DataFrame({
    "frame":frames,"t_sec":t_secs,"fps":fps,
    "flow_all_raw":flow_all_raw,"flow_upper_raw":flow_upper_raw,"flow_lower_raw":flow_lower_raw,
    "all_vx_raw":all_vx_raw,"all_vy_raw":all_vy_raw,
    "upper_vx_raw":upper_vx_raw,"upper_vy_raw":upper_vy_raw,
    "lower_vx_raw":lower_vx_raw,"lower_vy_raw":lower_vy_raw
})

mask_t0 = (df["t_sec"]>=T0_SEC); stats=[]
for key in ["flow_all_raw","flow_upper_raw","flow_lower_raw","all_vx_raw","all_vy_raw","upper_vx_raw","upper_vy_raw","lower_vx_raw","lower_vy_raw"]:
    x=df[key].to_numpy(); idx=np.where(mask_t0)[0]
    if idx.size>0:
        clean,st=series_clip_interp_medfilt(x[idx],fps,p1=PCLIP_LOW,p99=PCLIP_HIGH,max_gap_sec=MISS_MAX_SEC,median_win_fr=MEDIAN_WIN_FR)
        y=x.copy(); y[idx]=clean; df[key.replace("_raw","_clean")]=y; st["series"]=key.replace("_raw","_clean"); stats.append(st)

df.to_csv(csv_path,index=False); pd.DataFrame(stats).to_csv(stats_csv,index=False)
os.system(f"ffmpeg -y -i {out_try_path} -vcodec libx264 -pix_fmt yuv420p -profile:v baseline -level 3.0 -movflags +faststart {out_qt_path}")
print("[OK] Video:", out_qt_path); print("[OK] CSV:", csv_path); print("[OK] Stats:", stats_csv)
