"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║         AI TRAFFIC INTERSECTION INTELLIGENCE SYSTEM                            ║
║         Production-Grade Roundabout & Intersection Analytics                   ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Dev/Creator : tubakhxn                                                        ║
║  GitHub      : https://github.com/tubakhxn                                     ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  FEATURES                                                                      ║
║  • YOLOv8m vehicle detection (car/truck/bus/motorcycle)                        ║
║  • Colored zone overlays per approach (N/S/E/W) like ref image                ║
║  • Per-vehicle floating track IDs (numbers beside each car)                    ║
║  • Zone vehicle-count badges (large colored square per zone)                   ║
║  • Green trail dots + motion arrows on every tracked vehicle                   ║
║  • Congestion scoring, wrong-way detection, queue estimation                   ║
║  • Cinematic HUD: FPS, latency, counts, GPU mode, progress bar                ║
║  • Auto-install all deps, auto-download YOLO model                             ║
║                                                                                ║
║  USAGE                                                                         ║
║  python intersection_ai.py video.mp4                                          ║
║  python intersection_ai.py          (webcam)                                  ║
║                                                                                ║
║  CONTROLS: Q=Quit  P=Pause  S=Screenshot                                      ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

# ─── AUTO-INSTALL ─────────────────────────────────────────────────────────────
import sys, subprocess, importlib, os

_PKGS = {
    "cv2":         "opencv-python",
    "numpy":       "numpy",
    "torch":       "torch",
    "torchvision": "torchvision",
    "ultralytics": "ultralytics",
    "scipy":       "scipy",
    "PIL":         "pillow",
}

_missing = []
for mod, pip in _PKGS.items():
    try: importlib.import_module(mod)
    except ImportError: _missing.append(pip)

if _missing:
    print(f"\n📦  Installing: {', '.join(_missing)}")
    for p in _missing:
        print(f"    → {p}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
    print("✅  Done.\n")

# ─── IMPORTS ──────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import torch
import math
import time
import collections
from datetime import datetime
from ultralytics import YOLO

# ─── DEVICE ───────────────────────────────────────────────────────────────────
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = (DEVICE == "cuda")
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
print(f"\n  🖥️   Device : {GPU_NAME}")
print(f"  ⚡  FP16   : {USE_FP16}\n")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
OUTPUT_FILE      = "output_intersection_ai.mp4"
VEHICLE_CLASSES  = {2, 3, 5, 7}          # car, motorcycle, bus, truck
CONF_THRESH      = 0.28
IOU_THRESH       = 0.40
INFER_EVERY      = 1                      # run detection every N frames (1 = every frame)
TRAIL_LEN        = 28
MAX_LOST         = 25
FONT             = cv2.FONT_HERSHEY_SIMPLEX

# Zone definitions — 4 approach zones (fractions of frame W/H)
# Each zone = (x_frac_start, y_frac_start, x_frac_end, y_frac_end, name, color_BGRA)
# Colors match ref: North=blue, East=red/pink, South=green, West=gold
ZONE_DEFS = [
    ("NORTH", 0.25, 0.00, 0.75, 0.35,  (255, 140,  50, 80)),   # blue-ish
    ("EAST",  0.65, 0.25, 1.00, 0.75,  ( 80,  60, 220, 80)),   # red/pink
    ("SOUTH", 0.25, 0.65, 0.75, 1.00,  ( 60, 180,  60, 80)),   # green
    ("WEST",  0.00, 0.25, 0.35, 0.75,  ( 30, 180, 220, 80)),   # gold/orange
]

# HUD color palette
C_CYAN    = (255, 220,  40)
C_ORANGE  = (  0, 165, 255)
C_GREEN   = (  0, 230, 100)
C_PINK    = (200,  50, 255)
C_WHITE   = (240, 240, 255)
C_DIM     = (130, 140, 160)
C_PANEL   = (  8,  12,  22)
C_BORDER  = (  0, 200, 255)
C_WARN    = (  0,  60, 255)
C_TRAIL   = (  0, 220, 120)

ZONE_COLORS_BGR = {
    "NORTH": (255, 140,  50),
    "EAST":  ( 80,  60, 220),
    "SOUTH": ( 60, 200,  60),
    "WEST":  ( 30, 180, 220),
}

BADGE_COLORS = {
    "NORTH": (200,  80,  20),   # blue badge
    "EAST":  ( 40,  30, 180),   # red badge
    "SOUTH": ( 20, 140,  20),   # green badge
    "WEST":  ( 10, 120, 180),   # gold badge
}


# ─── TRACKER ──────────────────────────────────────────────────────────────────

class IoUTracker:
    """
    Lightweight IoU + proximity tracker.
    Maintains persistent numeric IDs, velocity, and trail history.
    """

    def __init__(self, max_lost=MAX_LOST, iou_thresh=0.18, dist_thresh=90):
        self.tracks     = {}       # id -> track_dict
        self.next_id    = 1
        self.max_lost   = max_lost
        self.iou_thresh = iou_thresh
        self.dist_thresh = dist_thresh

    @staticmethod
    def _iou(a, b):
        ax1,ay1,ax2,ay2 = a;  bx1,by1,bx2,by2 = b
        ix1,iy1 = max(ax1,bx1), max(ay1,by1)
        ix2,iy2 = min(ax2,bx2), min(ay2,by2)
        iw = max(0, ix2-ix1); ih = max(0, iy2-iy1)
        inter = iw*ih
        if inter == 0: return 0.0
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections):
        """
        detections: list of (x1,y1,x2,y2, conf, cls_id)
        Returns active track dict.
        """
        active = set()

        if not detections:
            for tid in list(self.tracks):
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]
            return {}

        det_boxes = [d[:4] for d in detections]
        tids = list(self.tracks.keys())

        matched_t = set(); matched_d = set()

        # Build score matrix: prefer IoU, fallback to distance
        pairs = []
        for ti, tid in enumerate(tids):
            tc = self.tracks[tid]
            tcx = (tc["box"][0]+tc["box"][2])//2
            tcy = (tc["box"][1]+tc["box"][3])//2
            for di, db in enumerate(det_boxes):
                iou  = self._iou(tc["box"], db)
                dcx  = (db[0]+db[2])//2
                dcy  = (db[1]+db[3])//2
                dist = math.hypot(dcx-tcx, dcy-tcy)
                # Combined score: IoU + proximity bonus
                score = iou + max(0, (self.dist_thresh - dist) / self.dist_thresh) * 0.4
                if score > 0.15:
                    pairs.append((score, ti, di))

        pairs.sort(reverse=True)
        for score, ti, di in pairs:
            tid = tids[ti]
            if ti in matched_t or di in matched_d: continue
            matched_t.add(ti); matched_d.add(di)
            self._update_track(tid, detections[di])
            active.add(tid)

        # Spawn new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di in matched_d: continue
            tid = self.next_id; self.next_id += 1
            box = det[:4]
            cx  = (box[0]+box[2])//2; cy = (box[1]+box[3])//2
            self.tracks[tid] = {
                "box":   box,
                "conf":  det[4],
                "cls":   det[5],
                "lost":  0,
                "age":   1,
                "vel":   (0.0, 0.0),
                "trail": collections.deque(maxlen=TRAIL_LEN),
                "color": self._id_color(tid),
            }
            self.tracks[tid]["trail"].append((cx, cy))
            active.add(tid)

        # Age lost tracks
        for tid in list(self.tracks):
            if tid not in active:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]

        return {tid: self.tracks[tid] for tid in active if tid in self.tracks}

    def _update_track(self, tid, det):
        t   = self.tracks[tid]
        box = det[:4]
        cx  = (box[0]+box[2])//2; cy = (box[1]+box[3])//2
        ocx = (t["box"][0]+t["box"][2])//2
        ocy = (t["box"][1]+t["box"][3])//2
        vx  = 0.65*t["vel"][0] + 0.35*(cx-ocx)
        vy  = 0.65*t["vel"][1] + 0.35*(cy-ocy)
        t["vel"]  = (vx, vy)
        t["box"]  = box
        t["conf"] = det[4]
        t["cls"]  = det[5]
        t["lost"] = 0
        t["age"]  += 1
        t["trail"].append((cx, cy))

    @staticmethod
    def _id_color(tid):
        hues = [100, 30, 200, 60, 160, 10, 140, 80, 50, 180, 20, 220]
        h    = hues[tid % len(hues)]
        sat  = 200 + (tid % 4)*13
        val  = 240
        hsv  = np.uint8([[[h, sat, val]]])
        bgr  = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


# ─── ZONE MANAGER ─────────────────────────────────────────────────────────────

class ZoneManager:
    def __init__(self, W, H):
        self.W = W; self.H = H
        # Convert fraction-based zones to pixel coords
        self.zones = []
        for name, xf1, yf1, xf2, yf2, color in ZONE_DEFS:
            x1 = int(xf1*W); y1 = int(yf1*H)
            x2 = int(xf2*W); y2 = int(yf2*H)
            self.zones.append({
                "name":  name,
                "x1":x1,"y1":y1,"x2":x2,"y2":y2,
                "color": color[:3],   # BGR
                "alpha": color[3]/255.0,
                "count": 0,
            })

    def assign(self, tracks):
        """Count vehicles per zone and tag each track with its zone."""
        for z in self.zones: z["count"] = 0
        for tid, t in tracks.items():
            cx = (t["box"][0]+t["box"][2])//2
            cy = (t["box"][1]+t["box"][3])//2
            t["zone"] = None
            for z in self.zones:
                if z["x1"] <= cx <= z["x2"] and z["y1"] <= cy <= z["y2"]:
                    z["count"] += 1
                    t["zone"]  = z["name"]
                    break

    def draw_zones(self, frame):
        """Draw colored semi-transparent zone overlays."""
        overlay = frame.copy()
        for z in self.zones:
            cv2.rectangle(overlay, (z["x1"],z["y1"]), (z["x2"],z["y2"]),
                          z["color"], -1)
        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

        # Zone border lines
        for z in self.zones:
            col = z["color"]
            bright = tuple(min(255, int(c*1.6)) for c in col)
            cv2.rectangle(frame, (z["x1"],z["y1"]), (z["x2"],z["y2"]),
                          bright, 2, cv2.LINE_AA)

    def draw_badges(self, frame):
        """Draw large count badges per zone like the reference image."""
        for z in self.zones:
            # Badge position: near top-center of each zone
            bx = (z["x1"]+z["x2"])//2 - 28
            by = z["y1"] + 14
            bw = 58; bh = 44
            # Make sure badge is inside frame
            bx = max(4, min(frame.shape[1]-bw-4, bx))
            by = max(4, min(frame.shape[0]-bh-4, by))

            col    = BADGE_COLORS[z["name"]]
            bright = tuple(min(255, int(c*1.5)) for c in col)

            # Filled badge
            ov = frame.copy()
            cv2.rectangle(ov, (bx,by), (bx+bw,by+bh), col, -1)
            cv2.addWeighted(ov, 0.88, frame, 0.12, 0, frame)
            cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), bright, 2, cv2.LINE_AA)

            # Count number (large)
            count_str = str(z["count"])
            (tw, th), _ = cv2.getTextSize(count_str, FONT, 1.1, 2)
            tx = bx + (bw-tw)//2; ty = by + (bh+th)//2 - 2
            cv2.putText(frame, count_str, (tx+1,ty+1), FONT, 1.1, (0,0,0),   3, cv2.LINE_AA)
            cv2.putText(frame, count_str, (tx,  ty),   FONT, 1.1, (255,255,255), 2, cv2.LINE_AA)

            # Zone name label below badge
            _tshadow(frame, z["name"], (bx, by+bh+14), 0.36,
                     tuple(min(255,int(c*1.7)) for c in col))


# ─── DRAWING UTILS ────────────────────────────────────────────────────────────

def _tshadow(img, text, pos, scale, color, thick=1):
    x, y = pos
    cv2.putText(img, text, (x+1,y+1), FONT, scale, (0,0,0),  thick+1, cv2.LINE_AA)
    cv2.putText(img, text, (x,  y),   FONT, scale, color,    thick,   cv2.LINE_AA)

def _draw_trail(img, trail):
    pts = list(trail)
    if len(pts) < 2: return
    for i in range(1, len(pts)):
        a = i / len(pts)
        col = (int(C_TRAIL[0]*a), int(C_TRAIL[1]*a), int(C_TRAIL[2]*a))
        cv2.circle(img, pts[i], max(1, int(a*4)), col, -1, cv2.LINE_AA)

def _draw_vehicle(frame, track, tid):
    x1, y1, x2, y2 = track["box"]
    col   = track["color"]
    conf  = track["conf"]
    cls   = track["cls"]
    zone  = track.get("zone", None)
    age   = track["age"]

    # Bright version for box edges
    bright = tuple(min(255, int(c*1.4)) for c in col)

    # ── Semi-transparent fill ──────────────────────────────────────────────
    sub = frame[y1:y2, x1:x2]
    if sub.size > 0:
        fill = np.full_like(sub, col, dtype=np.uint8)
        cv2.addWeighted(fill, 0.10, sub, 0.90, 0, sub)
        frame[y1:y2, x1:x2] = sub

    # ── Box with corner accents ────────────────────────────────────────────
    cv2.rectangle(frame, (x1,y1), (x2,y2), col, 1, cv2.LINE_AA)
    # Corner ticks
    L = 10
    for px, py, sx, sy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame,(px,py),(px+sx*L,py),    bright,2,cv2.LINE_AA)
        cv2.line(frame,(px,py),(px,py+sy*L),    bright,2,cv2.LINE_AA)

    # ── Trail ──────────────────────────────────────────────────────────────
    _draw_trail(frame, track["trail"])

    # ── Velocity arrow ─────────────────────────────────────────────────────
    cx = (x1+x2)//2; cy = (y1+y2)//2
    vx, vy = track["vel"]
    spd = math.hypot(vx, vy)
    if spd > 0.8:
        scale = 7.0
        ex = int(cx + vx*scale); ey = int(cy + vy*scale)
        cv2.arrowedLine(frame,(cx,cy),(ex,ey),C_TRAIL,2,cv2.LINE_AA,tipLength=0.45)

    # ── Floating ID badge (like ref: number beside car) ────────────────────
    id_str = str(tid)
    (tw, th), _ = cv2.getTextSize(id_str, FONT, 0.55, 1)
    bx1 = x2 + 3; by1 = y1
    bx2 = bx1 + tw + 8; by2 = by1 + th + 8
    # Clamp to frame
    fw_ = frame.shape[1]; fh_ = frame.shape[0]
    if bx2 > fw_: bx1 = x1 - tw - 11; bx2 = x1 - 3
    if by2 > fh_: by1 = y2 - th - 8;  by2 = y2

    ov2 = frame.copy()
    cv2.rectangle(ov2,(bx1,by1),(bx2,by2), col, -1)
    cv2.addWeighted(ov2, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame,(bx1,by1),(bx2,by2),bright,1,cv2.LINE_AA)
    cv2.putText(frame, id_str, (bx1+4,by2-4), FONT, 0.55, (255,255,255), 1, cv2.LINE_AA)

    # ── Confidence dot ─────────────────────────────────────────────────────
    cv2.circle(frame,(cx,cy),3,bright,-1,cv2.LINE_AA)


# ─── HUD ──────────────────────────────────────────────────────────────────────

class HUD:
    def __init__(self, W, H):
        self.W = W; self.H = H
        self.fps_buf = collections.deque(maxlen=30)

    def draw(self, frame, fps, latency_ms, tracks, zones, frame_idx,
             total_frames, elapsed, paused, wrong_way):
        W, H = self.W, self.H
        self.fps_buf.append(fps)
        avg_fps = float(np.mean(self.fps_buf))

        count    = len(tracks)
        total_v  = count
        cs       = min(100, total_v * 7)
        cs_col   = (C_WARN if cs>60 else (C_ORANGE if cs>30 else C_GREEN))
        cs_lbl   = ("CONGESTED" if cs>60 else ("MODERATE" if cs>30 else "FREE FLOW"))
        eff      = 100 - cs

        # ── Top banner ────────────────────────────────────────────────────
        ov = frame.copy()
        cv2.rectangle(ov,(0,0),(W,52),C_PANEL,-1)
        cv2.addWeighted(ov,0.85,frame,0.15,0,frame)
        cv2.line(frame,(0,52),(W,52),C_BORDER,1)

        _tshadow(frame,"◈  AI TRAFFIC INTERSECTION INTELLIGENCE SYSTEM",
                 (14,20),0.62,C_BORDER,1)
        _tshadow(frame,f"DEV: tubakhxn  |  github.com/tubakhxn",
                 (14,42),0.36,(80,180,200),1)
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        _tshadow(frame,ts,(W-210,20),0.43,C_DIM,1)
        _tshadow(frame,f"FPS: {avg_fps:.1f}   LAT: {latency_ms:.0f}ms   {GPU_NAME[:22]}",
                 (W-340,42),0.36,C_DIM,1)

        # ── Left analytics panel ──────────────────────────────────────────
        px,py,pw,ph = 10, 60, 200, 310
        ov2 = frame.copy()
        cv2.rectangle(ov2,(px,py),(px+pw,py+ph),C_PANEL,-1)
        cv2.addWeighted(ov2,0.82,frame,0.18,0,frame)
        cv2.rectangle(frame,(px,py),(px+pw,py+ph),C_BORDER,1)

        _tshadow(frame,"[ INTERSECTION ANALYTICS ]",(px+8,py+16),0.38,C_BORDER,1)
        cv2.line(frame,(px+6,py+22),(px+pw-6,py+22),C_BORDER,1)

        zone_counts  = {z["name"]:z["count"] for z in zones}
        rows = [
            ("VEHICLES",       f"{total_v}",             C_WHITE),
            ("ACTIVE TRACKS",  f"{len(tracks)}",          C_CYAN),
            ("WRONG-WAY",      f"{wrong_way}",            C_WARN if wrong_way else C_DIM),
            ("CONGESTION",     f"{cs:.0f}%",              cs_col),
            ("EFFICIENCY",     f"{eff:.0f}%",             C_GREEN),
            ("── ZONES ──",    "",                        C_DIM),
            ("  NORTH",        f"{zone_counts.get('NORTH',0)}",  ZONE_COLORS_BGR["NORTH"]),
            ("  EAST",         f"{zone_counts.get('EAST',0)}",   ZONE_COLORS_BGR["EAST"]),
            ("  SOUTH",        f"{zone_counts.get('SOUTH',0)}",  ZONE_COLORS_BGR["SOUTH"]),
            ("  WEST",         f"{zone_counts.get('WEST',0)}",   ZONE_COLORS_BGR["WEST"]),
        ]
        for i, (k, v, c) in enumerate(rows):
            y_row = py + 38 + i*26
            _tshadow(frame, k, (px+10, y_row),    0.34, C_DIM)
            if v: _tshadow(frame, v, (px+10, y_row+14), 0.46, c, 1)

        # Congestion badge
        bx = px+10; by = py+ph-42
        ov3 = frame.copy()
        cv2.rectangle(ov3,(bx,by),(bx+178,by+32),cs_col,-1)
        cv2.addWeighted(ov3,0.35,frame,0.65,0,frame)
        cv2.rectangle(frame,(bx,by),(bx+178,by+32),cs_col,2)
        _tshadow(frame, f"● {cs_lbl}", (bx+10,by+22), 0.55, cs_col, 2)

        # ── Bottom progress bar ───────────────────────────────────────────
        bar_y = H - 34
        ov4 = frame.copy()
        cv2.rectangle(ov4,(0,bar_y-2),(W,H),C_PANEL,-1)
        cv2.addWeighted(ov4,0.85,frame,0.15,0,frame)
        cv2.line(frame,(0,bar_y-2),(W,bar_y-2),C_BORDER,1)

        prog = frame_idx/max(total_frames,1)
        bw_  = int((W-4)*prog)
        cv2.rectangle(frame,(2,bar_y+4),(bw_+2,bar_y+12),C_BORDER,-1)
        cv2.rectangle(frame,(2,bar_y+4),(W-2,bar_y+12),(40,60,80),1)

        eta = (elapsed/max(prog,0.001))*(1-prog) if prog>0 else 0
        _tshadow(frame,f"PROGRESS: {prog*100:.1f}%  |  ETA: {int(eta)}s  |  FRAME {frame_idx}/{total_frames}",
                 (10,bar_y+28),0.40,C_DIM,1)

        # ── Paused overlay ────────────────────────────────────────────────
        if paused:
            _tshadow(frame,"⏸  PAUSED",(W//2-70,H//2),0.9,(0,120,255),2)

        return frame


# ─── HEATMAP ──────────────────────────────────────────────────────────────────

class Heatmap:
    def __init__(self, W, H):
        self.map = np.zeros((H, W), dtype=np.float32)

    def update(self, tracks):
        self.map *= 0.985
        for t in tracks.values():
            cx = (t["box"][0]+t["box"][2])//2
            cy = (t["box"][1]+t["box"][3])//2
            y1 = max(0,cy-18); y2 = min(self.map.shape[0],cy+18)
            x1 = max(0,cx-18); x2 = min(self.map.shape[1],cx+18)
            self.map[y1:y2,x1:x2] += 1.2
        np.clip(self.map, 0, 12, out=self.map)

    def blend(self, frame):
        if self.map.max() < 0.1: return frame
        norm   = np.clip(self.map / (self.map.max()+1e-6), 0, 1)
        colored= cv2.applyColorMap((norm*255).astype(np.uint8), cv2.COLORMAP_HOT)
        mask   = (norm > 0.12).astype(np.float32)[...,None]
        alpha  = norm[...,None] * 0.30 * mask
        frame  = (frame.astype(np.float32)*(1-alpha) + colored.astype(np.float32)*alpha)
        return frame.astype(np.uint8)


# ─── WRONG-WAY DETECTOR ───────────────────────────────────────────────────────

class WrongWayDetector:
    """
    Flags vehicles moving in the opposite expected direction
    through each zone approach (inbound should move toward center).
    """
    def __init__(self, W, H):
        self.W = W; self.H = H
        self.cx = W//2; self.cy = H//2
        self.count = 0
        self.flags = set()

    def check(self, tracks):
        newly_flagged = set()
        for tid, t in tracks.items():
            vx, vy = t["vel"]
            spd = math.hypot(vx,vy)
            if spd < 1.2: continue
            cx = (t["box"][0]+t["box"][2])//2
            cy = (t["box"][1]+t["box"][3])//2
            zone = t.get("zone")
            wrong = False
            if zone == "NORTH" and cy < self.cy and vy < -2:   # going away from center northward
                wrong = True
            elif zone == "SOUTH" and cy > self.cy and vy > 2:
                wrong = True
            elif zone == "EAST"  and cx > self.cx and vx > 2:
                wrong = True
            elif zone == "WEST"  and cx < self.cx and vx < -2:
                wrong = True
            if wrong:
                newly_flagged.add(tid)
                if tid not in self.flags:
                    self.count += 1
        self.flags = newly_flagged
        return newly_flagged

    def draw_warnings(self, frame, tracks, flagged):
        for tid in flagged:
            if tid not in tracks: continue
            x1,y1,x2,y2 = tracks[tid]["box"]
            cv2.rectangle(frame,(x1-3,y1-3),(x2+3,y2+3),C_WARN,3,cv2.LINE_AA)
            _tshadow(frame,"⚠ WRONG-WAY",(x1,y2+16),0.50,C_WARN,2)


# ─── MAIN PROCESSING LOOP ─────────────────────────────────────────────────────

def run(source):
    # ── Open source ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"   Cannot open: {source}")
        sys.exit(1)

    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS_IN      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    TOTAL_FRAMES= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 99999
    print(f"\n   Source : {source}  |  {W}×{H}  {FPS_IN:.1f}fps  {TOTAL_FRAMES} frames")

    # ── Output ────────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS_IN, (W,H))
    print(f"  Output : {OUTPUT_FILE}\n")

    # ── Load YOLO ─────────────────────────────────────────────────────────
    print("    Loading YOLOv8m detection model...")
    model = YOLO("yolov8m.pt")
    model.to(DEVICE)
    if USE_FP16: model.model.half()
    print("   Model ready.\n")

    # ── Subsystems ────────────────────────────────────────────────────────
    tracker  = IoUTracker()
    zones    = ZoneManager(W, H)
    heatmap  = Heatmap(W, H)
    hud      = HUD(W, H)
    wwd      = WrongWayDetector(W, H)

    frame_idx  = 0
    paused     = False
    prev_time  = time.time()
    start_time = time.time()
    last_dets  = []

    print("   Running. Controls: Q=Quit  P=Pause  S=Screenshot\n")
    print("═"*62)

    while True:
        if paused:
            k = cv2.waitKey(30) & 0xFF
            if   k == ord('q'): break
            elif k == ord('p'): paused = False
            continue

        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        t0 = time.time()

        # ── Detection (every INFER_EVERY frames) ──────────────────────────
        if frame_idx % INFER_EVERY == 0:
            results  = model.predict(
                frame,
                conf    = CONF_THRESH,
                iou     = IOU_THRESH,
                verbose = False,
                device  = DEVICE,
                half    = USE_FP16,
                imgsz   = 640,
            )
            last_dets = []
            for r in results:
                if r.boxes is None: continue
                for box, conf, cls in zip(
                    r.boxes.xyxy.cpu().numpy(),
                    r.boxes.conf.cpu().numpy(),
                    r.boxes.cls.cpu().numpy().astype(int)
                ):
                    if int(cls) not in VEHICLE_CLASSES: continue
                    x1,y1,x2,y2 = map(int,box)
                    last_dets.append((x1,y1,x2,y2,float(conf),int(cls)))

        # ── Track ─────────────────────────────────────────────────────────
        active = tracker.update(last_dets)

        # ── Analytics ─────────────────────────────────────────────────────
        zones.assign(active)
        heatmap.update(active)
        flagged = wwd.check(active)

        # ── Timing ────────────────────────────────────────────────────────
        now      = time.time()
        latency  = (now - t0)*1000
        fps      = 1.0/max(now-prev_time,1e-6)
        prev_time= now
        elapsed  = now - start_time

        # ══════════════════════════════════════════════════════════════════
        # RENDER PIPELINE
        # ══════════════════════════════════════════════════════════════════

        out = frame.copy()

        # 1. Heatmap blend (subtle background warmth)
        out = heatmap.blend(out)

        # 2. Zone overlays (colored semi-transparent regions)
        zones.draw_zones(out)

        # 3. Per-vehicle: trails, boxes, ID badges
        for tid, t in active.items():
            _draw_vehicle(out, t, tid)

        # 4. Wrong-way warnings on top
        wwd.draw_warnings(out, active, flagged)

        # 5. Zone count badges (large number per zone)
        zones.draw_badges(out)

        # 6. HUD panels + progress bar
        out = hud.draw(out, fps, latency, active, zones.zones,
                       frame_idx, TOTAL_FRAMES, elapsed, paused, wwd.count)

        # ── Output ────────────────────────────────────────────────────────
        writer.write(out)
        cv2.imshow("AI Traffic Intersection Intelligence — tubakhxn", out)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): break
        elif key == ord('p'): paused = True; print("\n  ⏸  Paused")
        elif key == ord('s'):
            ss = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(ss, out)
            print(f"\n  📸  Saved: {ss}")

        # Progress to terminal
        if frame_idx % 30 == 0:
            p   = frame_idx/TOTAL_FRAMES*100
            eta = elapsed/max(p/100,0.001)*(1-p/100)
            bar = "█"*int(p//4) + "░"*(25-int(p//4))
            print(f"\r  [{bar}] {p:5.1f}%  FPS:{fps:5.1f}  "
                  f"Vehicles:{len(active)}  WrongWay:{wwd.count}  ETA:{int(eta)}s   ",
                  end="", flush=True)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\n\n  ✅  Done!  Output → {OUTPUT_FILE}\n")


# ─── ENTRY ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    try: src = int(src)
    except (ValueError, TypeError): pass

    print(f"""
╔══════════════════════════════════════════════════════════╗
║      AI TRAFFIC INTERSECTION INTELLIGENCE SYSTEM        ║
║            tubakhxn  |  github.com/tubakhxn             ║
╚══════════════════════════════════════════════════════════╝
  Source  : {src}
  Device  : {GPU_NAME}
  Output  : {OUTPUT_FILE}
""")
    run(src)