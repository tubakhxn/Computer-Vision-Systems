"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║          WILDLIFE & BIRD POPULATION MONITORING                          ║
║          Real-Time Ecological Intelligence System                               ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Dev/Creator : tubakhxn                                                        ║
║  GitHub      : https://github.com/tubakhxn                                     ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  ADDED: tqdm progress bar for video processing                                 ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

# ─── AUTO-INSTALL DEPENDENCIES ───────────────────────────────────────────────
import sys, subprocess, importlib, os

REQUIRED = {
    "ultralytics": "ultralytics",
    "cv2":         "opencv-python",
    "numpy":       "numpy",
    "torch":       "torch",
    "torchvision": "torchvision",
    "scipy":       "scipy",
    "PIL":         "pillow",
    "matplotlib":  "matplotlib",
    "supervision": "supervision",
    "tqdm":        "tqdm",
}

def _print_bar(label: str, percent: int, width: int = 40):
    filled = int(width * percent / 100)
    bar    = "█" * filled + "░" * (width - filled)
    print(f"\r   {label}  [{bar}] {percent:3d}%", end="", flush=True)

def _install_with_progress(pip_name: str):
    import re
    print(f"\n   📦  Installing {pip_name}...")
    _print_bar(pip_name, 0)
    cmd = [sys.executable, "-m", "pip", "install", "--progress-bar", "on", "-q", pip_name]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    last_pct = 0
    for line in proc.stdout:
        line = line.strip()
        m = re.search(r'(\d+)%', line)
        if m:
            pct = min(int(m.group(1)), 99)
            if pct > last_pct:
                last_pct = pct
                _print_bar(pip_name, pct)
        m2 = re.search(r'([\d.]+)\s*(k?B|MB)\s*/\s*([\d.]+)\s*(MB|GB)', line, re.I)
        if m2:
            def to_mb(val, unit):
                v = float(val); u = unit.lower()
                if u in ("kb","b"): return v/1024 if u=="kb" else v/1048576
                if u == "gb": return v*1024
                return v
            downloaded = to_mb(m2.group(1), m2.group(2))
            total      = to_mb(m2.group(3), m2.group(4))
            if total > 0:
                pct = min(int(downloaded/total*100), 99)
                if pct > last_pct:
                    last_pct = pct
                    _print_bar(pip_name, pct)
    proc.wait()
    _print_bar(pip_name, 100)
    print()

needs_install = []
for mod, pip_name in REQUIRED.items():
    try:
        importlib.import_module(mod)
    except ImportError:
        needs_install.append(pip_name)

if needs_install:
    print(f"\n📦  {len(needs_install)} package(s) to install: {', '.join(needs_install)}")
    for p in needs_install:
        _install_with_progress(p)
    print("✅  All dependencies installed.\n")

# ─── IMPORTS ─────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import torch
import time
import math
import collections
from datetime import datetime
from scipy.ndimage import gaussian_filter
from ultralytics import YOLO
from tqdm import tqdm

# ─── CONSTANTS & CONFIG ───────────────────────────────────────────────────────

C_NEON_GREEN   = (0,   255, 120)
C_NEON_CYAN    = (0,   230, 255)
C_NEON_ORANGE  = (0,   165, 255)
C_NEON_PINK    = (180,  0,  255)
C_NEON_YELLOW  = (0,   255, 230)
C_WHITE        = (255, 255, 255)
C_DIM_WHITE    = (180, 180, 180)
C_BLACK        = (0,   0,   0)
C_ACCENT_BLUE  = (255, 160,  30)
C_TRACK_BASE   = (50,  200, 120)

TRAIL_LENGTH   = 40
HEATMAP_DECAY  = 0.97
HEATMAP_SIGMA  = 22
FONT           = cv2.FONT_HERSHEY_SIMPLEX

WILDLIFE_CLASSES = {14: "bird", 15: "cat", 16: "dog", 17: "horse",
                    18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
                    22: "zebra", 23: "giraffe"}

BIRD_CLASS_ID  = 14

# ─── STARTUP BANNER ──────────────────────────────────────────────────────────
def print_banner():
    banner = r"""
╔══════════════════════════════════════════════════════════════╗
║   WILDLIFE & BIRD POPULATION MONITORING AI                  ║
║   REAL-TIME ECOLOGICAL INTELLIGENCE SYSTEM                  ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

    startup_steps = [
        ("🦅  Loading wildlife detection models",   0.02),
        ("📊  Initializing population analytics",   0.01),
        ("🧭  Starting migration intelligence",     0.01),
        ("🌍  Launching ecological platform",       0.01),
    ]

    for label, delay in startup_steps:
        with tqdm(
            total=100,
            desc=f"   {label}",
            bar_format="{desc}  |{bar}| {percentage:3.0f}%",
            ncols=72,
            colour="green",
            ascii="░█",
        ) as pbar:
            for i in range(10):
                time.sleep(delay)
                pbar.update(10)

    print("\n" + "═"*64 + "\n")

# ─── MODEL DOWNLOAD PROGRESS ─────────────────────────────────────────────────

def load_model_with_progress(model_name: str, device: str, use_half: bool):
    import urllib.request
    from pathlib import Path

    weights_path = Path(model_name)
    cache_dir    = Path.home() / ".cache" / "ultralytics"
    cached       = cache_dir / model_name

    if not weights_path.exists() and not cached.exists():
        url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}"
        print(f"\n   ⬇️   Downloading {model_name} weights...")
        with urllib.request.urlopen(url) as response:
            total = int(response.headers.get("Content-Length", 0))
            chunk = 1024 * 64
            with tqdm(
                total=total, unit="B", unit_scale=True, unit_divisor=1024,
                desc=f"   {model_name}",
                bar_format="{desc}  |{bar}| {percentage:3.0f}%  {n_fmt}/{total_fmt}  {rate_fmt}",
                ncols=72, colour="cyan", ascii="░█",
            ) as pbar:
                with open(model_name, "wb") as f:
                    while True:
                        data = response.read(chunk)
                        if not data: break
                        f.write(data)
                        pbar.update(len(data))
        print(f"   ✅  {model_name} downloaded.\n")

    print(f"   🔧  Loading model into {device.upper()}...")
    with tqdm(
        total=4, desc="   Model init",
        bar_format="{desc}  |{bar}| {percentage:3.0f}%  step {n}/{total}",
        ncols=72, colour="yellow", ascii="░█",
    ) as pbar:
        model = YOLO(model_name);        pbar.update(1)
        model.to(device);                pbar.update(1)
        if use_half:
            model.model.half()
        pbar.update(1)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model.predict(dummy, verbose=False, device=device, imgsz=640)
        pbar.update(1)

    return model

# ─── SIMPLE BYTE-TRACK-STYLE TRACKER ─────────────────────────────────────────

class SimpleTracker:
    def __init__(self, max_lost=30, iou_threshold=0.25):
        self.max_lost    = max_lost
        self.iou_thresh  = iou_threshold
        self.tracks      = {}
        self.next_id     = 1
        self.frame_count = 0

    @staticmethod
    def _iou(a, b):
        ax1,ay1,ax2,ay2 = a
        bx1,by1,bx2,by2 = b
        ix1,iy1 = max(ax1,bx1), max(ay1,by1)
        ix2,iy2 = min(ax2,bx2), min(ay2,by2)
        iw,ih   = max(0,ix2-ix1), max(0,iy2-iy1)
        inter   = iw*ih
        if inter == 0: return 0.0
        ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
        return inter/ua if ua>0 else 0.0

    def update(self, detections):
        self.frame_count += 1
        active_ids = set()

        if not detections:
            for tid in list(self.tracks):
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]
            return {}

        det_boxes  = [d[:4] for d in detections]
        track_ids  = list(self.tracks.keys())
        matched_det = set()
        matched_trk = set()

        pairs = []
        for ti, tid in enumerate(track_ids):
            for di, db in enumerate(det_boxes):
                iou = self._iou(self.tracks[tid]["box"], db)
                if iou > self.iou_thresh:
                    pairs.append((iou, ti, di))
        pairs.sort(reverse=True)

        for iou, ti, di in pairs:
            tid = track_ids[ti]
            if ti in matched_trk or di in matched_det: continue
            matched_trk.add(ti); matched_det.add(di)
            self._update_track(tid, detections[di])
            active_ids.add(tid)

        for di, det in enumerate(detections):
            if di not in matched_det:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {
                    "box": det[:4], "conf": det[4], "cls": det[5],
                    "lost": 0, "age": 1,
                    "trail": collections.deque(maxlen=TRAIL_LENGTH),
                    "vel": (0.0, 0.0),
                    "color": self._id_color(tid),
                }
                cx = (det[0]+det[2])//2; cy = (det[1]+det[3])//2
                self.tracks[tid]["trail"].append((cx, cy))
                active_ids.add(tid)

        for tid in list(self.tracks):
            if tid not in active_ids:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]

        return {tid: self.tracks[tid] for tid in active_ids if tid in self.tracks}

    def _update_track(self, tid, det):
        box = det[:4]
        cx = (box[0]+box[2])//2; cy = (box[1]+box[3])//2
        t  = self.tracks[tid]
        old_cx = (t["box"][0]+t["box"][2])//2
        old_cy = (t["box"][1]+t["box"][3])//2
        vx = 0.7*t["vel"][0] + 0.3*(cx-old_cx)
        vy = 0.7*t["vel"][1] + 0.3*(cy-old_cy)
        t["vel"] = (vx, vy); t["box"] = box; t["conf"] = det[4]
        t["cls"] = det[5];   t["lost"] = 0;  t["age"] += 1
        t["trail"].append((cx, cy))

    @staticmethod
    def _id_color(tid):
        hues = [0,30,60,90,120,150,180,210,240,270,300,330]
        h    = hues[tid % len(hues)]
        hsv  = np.uint8([[[h, 220, 255]]])
        bgr  = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


# ─── ANALYTICS ENGINE ────────────────────────────────────────────────────────

class AnalyticsEngine:
    def __init__(self, w, h):
        self.W, self.H = w, h
        self.heatmap   = np.zeros((h, w), dtype=np.float32)
        self.history   = collections.deque(maxlen=120)
        self.pop_max   = 1

    def update(self, tracks):
        frame_data = {"count": len(tracks), "centers": [], "velocities": []}
        self.heatmap *= HEATMAP_DECAY
        for tid, t in tracks.items():
            cx = (t["box"][0]+t["box"][2])//2
            cy = (t["box"][1]+t["box"][3])//2
            frame_data["centers"].append((cx, cy))
            frame_data["velocities"].append(t["vel"])
            x1 = max(0, cx-30); x2 = min(self.W, cx+30)
            y1 = max(0, cy-30); y2 = min(self.H, cy+30)
            self.heatmap[y1:y2, x1:x2] += 0.4
        self.heatmap = np.clip(self.heatmap, 0, 10)
        self.history.append(frame_data)
        if len(tracks) > self.pop_max: self.pop_max = len(tracks)

    def population_label(self, count):
        if   count == 0:  return "NO ACTIVITY",         C_DIM_WHITE
        elif count <  5:  return "LOW POPULATION",      C_NEON_GREEN
        elif count < 20:  return "MODERATE POPULATION", C_NEON_YELLOW
        elif count < 50:  return "HIGH POPULATION",     C_NEON_ORANGE
        else:             return "MASS FLOCK EVENT",    C_NEON_PINK

    def migration_direction(self, tracks):
        vels = [t["vel"] for t in tracks.values() if math.hypot(*t["vel"]) > 0.5]
        if len(vels) < 3: return "STATIONARY", 0.0
        avg_vx = np.mean([v[0] for v in vels])
        avg_vy = np.mean([v[1] for v in vels])
        angle  = math.degrees(math.atan2(-avg_vy, avg_vx)) % 360
        mag    = math.hypot(avg_vx, avg_vy)
        dirs   = {(315,360):"EASTBOUND",(0,45):"EASTBOUND",
                  (45,135):"SOUTHBOUND",(135,225):"WESTBOUND",(225,315):"NORTHBOUND"}
        label  = "EASTBOUND"
        for (lo,hi), name in dirs.items():
            if lo <= angle < hi: label = name; break
        return label, mag

    def flock_score(self, tracks):
        centers = [((t["box"][0]+t["box"][2])//2, (t["box"][1]+t["box"][3])//2)
                   for t in tracks.values()]
        if len(centers) < 2: return 0.0, 0.0
        arr      = np.array(centers, dtype=float)
        std      = np.std(arr, axis=0).mean()
        cohesion = max(0, 1.0 - std/max(self.W, self.H))
        vels = [t["vel"] for t in tracks.values() if math.hypot(*t["vel"]) > 0.3]
        sync = 0.0
        if len(vels) >= 2:
            angles    = [math.atan2(v[1], v[0]) for v in vels]
            angle_std = np.std(angles)
            sync      = max(0, 1.0 - angle_std/math.pi)
        return round(cohesion, 2), round(sync, 2)

    def habitat_score(self, tracks):
        if not self.history: return 0
        counts = [d["count"] for d in self.history]
        trend  = np.mean(counts[-10:]) if len(counts)>=10 else np.mean(counts)
        return min(100, int(trend/max(self.pop_max,1)*100))

    def render_heatmap(self, frame):
        blurred = gaussian_filter(self.heatmap, sigma=HEATMAP_SIGMA)
        if blurred.max() < 0.01: return frame
        norm    = np.clip(blurred/(blurred.max()+1e-6), 0, 1)
        colored = cv2.applyColorMap((norm*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        mask    = (norm > 0.05).astype(np.float32)
        alpha   = (norm * 0.55 * mask)[..., None]
        frame   = (frame.astype(np.float32)*(1-alpha) + colored.astype(np.float32)*alpha)
        return frame.astype(np.uint8)


# ─── DRAWING HELPERS ─────────────────────────────────────────────────────────

def _alpha_rect(img, x1,y1,x2,y2, color, alpha=0.35):
    sub = img[y1:y2, x1:x2]
    if sub.size == 0: return
    rect = np.full_like(sub, color, dtype=np.uint8)
    cv2.addWeighted(rect, alpha, sub, 1-alpha, 0, sub)
    img[y1:y2, x1:x2] = sub

def _corner_box(img, x1,y1,x2,y2, color, thickness=2, corner_len=14):
    pts = [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]
    for px,py,sx,sy in pts:
        cv2.line(img,(px,py),(px+sx*corner_len,py),color,thickness,cv2.LINE_AA)
        cv2.line(img,(px,py),(px,py+sy*corner_len),color,thickness,cv2.LINE_AA)

def _text_shadow(img, text, pos, scale, color, thickness=1):
    x,y = pos
    cv2.putText(img, text, (x+1,y+1), FONT, scale, (0,0,0),  thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, (x,  y),   FONT, scale, color,     thickness,   cv2.LINE_AA)

def _draw_trail(img, trail, color):
    pts = list(trail)
    if len(pts) < 2: return
    for i in range(1, len(pts)):
        alpha_f = i / len(pts)
        fade    = tuple(int(c * alpha_f) for c in color)
        thick   = max(1, int(alpha_f * 3))
        cv2.line(img, pts[i-1], pts[i], fade, thick, cv2.LINE_AA)


# ─── HUD RENDERER ─────────────────────────────────────────────────────────────

class HUDRenderer:
    def __init__(self, W, H, device_label):
        self.W = W; self.H = H
        self.device = device_label
        self.fps_history = collections.deque(maxlen=30)

    def render(self, frame, fps, latency_ms, tracks, analytics, frame_idx, paused):
        h, w = frame.shape[:2]
        count           = len(tracks)
        pop_lbl,pop_col = analytics.population_label(count)
        mig_dir,mig_mag = analytics.migration_direction(tracks)
        cohesion, sync  = analytics.flock_score(tracks)
        hab_score       = analytics.habitat_score(tracks)
        species_set     = set(t["cls"] for t in tracks.values())
        n_species       = len(species_set)
        n_birds         = sum(1 for t in tracks.values() if t["cls"]==BIRD_CLASS_ID)

        self.fps_history.append(fps)
        avg_fps = np.mean(self.fps_history) if self.fps_history else fps

        _alpha_rect(frame, 8, 8, 260, 230, (5,12,25), 0.78)
        cv2.rectangle(frame, (8,8), (260,230), C_NEON_CYAN, 1)
        y = 32
        _text_shadow(frame, "WILDLIFE MONITOR AI", (16,y), 0.45, C_NEON_CYAN, 1)
        y += 2
        cv2.line(frame,(16,y+2),(254,y+2), C_NEON_CYAN,1)
        y += 18

        rows = [
            (f"FPS   {avg_fps:5.1f}",           C_NEON_GREEN),
            (f"LATENCY  {latency_ms:.1f}ms",     C_NEON_GREEN if latency_ms<50 else C_NEON_ORANGE),
            (f"FRAME  #{frame_idx}",             C_DIM_WHITE),
            (f"MODE   {self.device}",            C_ACCENT_BLUE),
            ("───────────────────",              (60,60,80)),
            (f"BIRDS   {n_birds:>4d}",           C_NEON_YELLOW),
            (f"WILDLIFE {count:>4d}",            C_NEON_YELLOW),
            (f"TRACKS  {len(tracks):>4d}",       C_NEON_CYAN),
            (f"SPECIES {n_species:>4d}",         C_NEON_GREEN),
            ("───────────────────",              (60,60,80)),
            (f"HAB SCORE {hab_score:>3d}/100",   C_NEON_ORANGE),
            (f"COHESION  {cohesion:.2f}",        C_NEON_GREEN),
            (f"SYNC     {sync:.2f}",             C_NEON_CYAN),
        ]
        for txt, col in rows:
            _text_shadow(frame, txt, (18, y), 0.40, col, 1)
            y += 16

        rx1 = w-260; ry1 = 8; rx2 = w-8; ry2 = 160
        _alpha_rect(frame, rx1, ry1, rx2, ry2, (5,12,25), 0.78)
        cv2.rectangle(frame,(rx1,ry1),(rx2,ry2), C_NEON_ORANGE, 1)
        _text_shadow(frame,"MIGRATION INTEL",(rx1+10,ry1+22),0.42,C_NEON_ORANGE,1)
        cv2.line(frame,(rx1+8,ry1+26),(rx2-8,ry1+26),C_NEON_ORANGE,1)
        _text_shadow(frame, mig_dir, (rx1+10,ry1+50), 0.60,
                     C_NEON_PINK if "BOUND" in mig_dir else C_DIM_WHITE, 1)

        cx_c = rx1+200; cy_c = ry1+100; r_c = 38
        cv2.circle(frame,(cx_c,cy_c),r_c,(30,40,55),-1)
        cv2.circle(frame,(cx_c,cy_c),r_c,C_NEON_ORANGE,1)
        for label, dx, dy in [("N",0,-1),("S",0,1),("E",1,0),("W",-1,0)]:
            lx = cx_c+int(dx*(r_c-8)); ly = cy_c+int(dy*(r_c-8))
            cv2.putText(frame,label,(lx-5,ly+4),FONT,0.32,C_DIM_WHITE,1,cv2.LINE_AA)
        arrow_map = {"NORTHBOUND":(0,-1),"SOUTHBOUND":(0,1),
                     "EASTBOUND":(1,0),"WESTBOUND":(-1,0),"STATIONARY":(0,0)}
        adx, ady = arrow_map.get(mig_dir,(0,0))
        if mig_mag > 0.3:
            ax2_ = cx_c+int(adx*(r_c-10)); ay2_ = cy_c+int(ady*(r_c-10))
            cv2.arrowedLine(frame,(cx_c,cy_c),(ax2_,ay2_),C_NEON_PINK,2,cv2.LINE_AA,tipLength=0.35)
        _text_shadow(frame, f"MAG {mig_mag:.1f}", (rx1+10, ry1+78), 0.36, C_DIM_WHITE, 1)

        bar_h = 40
        _alpha_rect(frame, 0, h-bar_h, w, h, (3,8,18), 0.88)
        cv2.line(frame,(0,h-bar_h),(w,h-bar_h),C_NEON_CYAN,1)
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        _text_shadow(frame, ts, (12, h-14), 0.42, C_NEON_CYAN, 1)
        _text_shadow(frame, pop_lbl, (w//2-90, h-14), 0.55, pop_col, 1)
        _text_shadow(frame, "tubakhxn | github.com/tubakhxn", (w-280, h-14), 0.35, (80,80,100), 1)

        scan_y = (frame_idx * 3) % h
        cv2.line(frame, (0,scan_y), (w,scan_y), (0,200,100), 1)

        if paused:
            _alpha_rect(frame, w//2-80, h//2-20, w//2+80, h//2+20, (10,10,10), 0.8)
            _text_shadow(frame,"⏸ PAUSED",(w//2-60,h//2+8),0.75,C_NEON_PINK,2)

        return frame


# ─── MAIN MONITORING SYSTEM ──────────────────────────────────────────────────

def run(source=0):
    print_banner()

    if torch.cuda.is_available():
        device     = "cuda"
        device_lbl = f"GPU:{torch.cuda.get_device_name(0)[:18]}"
        use_half   = True
        print(f"    CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device     = "cpu"
        device_lbl = "CPU"
        use_half   = False
        print("   ℹ  Running on CPU (no CUDA detected)")

    model = load_model_with_progress("yolov8m.pt", device, use_half)
    print("    Model ready.\n")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"  Cannot open source: {source}")
        sys.exit(1)

    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_video     = total_frames > 0   # False for webcam / live stream

    print(f"   📹  Source : {source}  |  {W}×{H}  {FPS:.1f}fps")
    if is_video:
        mins = int(total_frames/FPS)//60
        secs = int(total_frames/FPS)%60
        print(f"     Frames : {total_frames}  (~{mins}m {secs}s)")

    out_path = "output_wildlife_monitoring_ai.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))
    print(f"     Output : {out_path}\n")

    tracker   = SimpleTracker(max_lost=30, iou_threshold=0.2)
    analytics = AnalyticsEngine(W, H)
    hud       = HUDRenderer(W, H, device_lbl)

    frame_idx = 0
    paused    = False
    prev_time = time.time()

    print("     Monitoring active. Press Q=Quit | P=Pause | S=Screenshot\n")
    print("═"*64)

    # ── VIDEO PROGRESS BAR (only for file input, not webcam) ─────────
    pbar = tqdm(
        total=total_frames if is_video else None,
        desc="   🦅  Processing",
        unit="frame",
        bar_format=(
            "{desc}  |{bar}| {percentage:3.0f}%  "
            "{n_fmt}/{total_fmt}  [{elapsed}<{remaining}  {rate_fmt}]"
            if is_video else
            "{desc}  {n_fmt} frames  [{elapsed}  {rate_fmt}]"
        ),
        ncols=80,
        colour="green",
        ascii="░█",
        dynamic_ncols=False,
    ) if is_video else tqdm(
        desc="   🦅  Live",
        unit="frame",
        bar_format="{desc}  {n_fmt} frames  [{elapsed}  {rate_fmt}]",
        ncols=60,
        colour="cyan",
        ascii="░█",
    )

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n   ⏹  Stream ended.")
                break
        else:
            key = cv2.waitKey(30) & 0xFF
            if   key == ord('q'): break
            elif key == ord('p'): paused = False
            continue

        t0 = time.time()
        frame_idx += 1

        results = model.predict(
            frame, conf=0.30, iou=0.40, verbose=False,
            device=device, half=use_half, imgsz=640,
        )

        detections = []
        for r in results:
            if r.boxes is None: continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clses = r.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls in zip(boxes, confs, clses):
                if cls in WILDLIFE_CLASSES:
                    x1,y1,x2,y2 = map(int, box)
                    detections.append((x1,y1,x2,y2, float(conf), int(cls)))

        active_tracks = tracker.update(detections)
        analytics.update(active_tracks)

        now       = time.time()
        latency   = (now - t0) * 1000
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        # Update progress bar with live stats
        n_detected = len(active_tracks)
        n_birds    = sum(1 for t in active_tracks.values() if t["cls"]==BIRD_CLASS_ID)
        pbar.set_postfix({
            "fps"    : f"{fps:.1f}",
            "birds"  : n_birds,
            "tracked": n_detected,
        }, refresh=False)
        pbar.update(1)

        out = analytics.render_heatmap(frame.copy())

        for tid, t in active_tracks.items():
            color      = t["color"]
            x1,y1,x2,y2 = t["box"]
            cls_id     = t["cls"]
            conf       = t["conf"]
            label      = WILDLIFE_CLASSES.get(cls_id, "animal")

            _draw_trail(out, t["trail"], color)
            _alpha_rect(out, x1,y1,x2,y2, color, alpha=0.08)

            box_col = (min(255,color[0]+60), min(255,color[1]+60), min(255,color[2]+60))
            _corner_box(out, x1,y1,x2,y2, box_col, thickness=2, corner_len=12)

            badge_w = 80; badge_h = 20
            bx1_ = x1; bx2_ = x1+badge_w
            by1_ = max(0, y1-badge_h-2); by2_ = max(0, y1-2)
            _alpha_rect(out, bx1_,by1_,bx2_,by2_, color, alpha=0.75)
            id_txt = f"#{tid} {label[:4].upper()} {conf:.2f}"
            cv2.putText(out, id_txt, (bx1_+3,by2_-4), FONT, 0.34, C_WHITE, 1, cv2.LINE_AA)

            cx_ = (x1+x2)//2; cy_ = (y1+y2)//2
            cv2.circle(out,(cx_,cy_),3, color, -1, cv2.LINE_AA)
            cv2.circle(out,(cx_,cy_),6, box_col, 1, cv2.LINE_AA)

            vx_, vy_ = t["vel"]
            if math.hypot(vx_, vy_) > 1.0:
                ex_ = int(cx_ + vx_*6); ey_ = int(cy_ + vy_*6)
                cv2.arrowedLine(out,(cx_,cy_),(ex_,ey_),C_NEON_YELLOW,1,cv2.LINE_AA,tipLength=0.4)

        overlay = out.copy()
        for gx in range(0, W, 120):
            cv2.line(overlay,(gx,0),(gx,H),(20,40,30),1)
        for gy in range(0, H, 120):
            cv2.line(overlay,(0,gy),(W,gy),(20,40,30),1)
        cv2.addWeighted(overlay, 0.20, out, 0.80, 0, out)

        out = hud.render(out, fps, latency, active_tracks, analytics, frame_idx, paused)

        writer.write(out)
        cv2.imshow("Wildlife & Bird Population Monitoring AI — tubakhxn", out)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): break
        elif key == ord('p'): paused = True
        elif key == ord('s'):
            ss_name = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(ss_name, out)
            print(f"\n   📸  Screenshot saved: {ss_name}")

    pbar.close()
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\n     Done. Output saved to: {out_path}")
    print("    Wildlife Monitoring Session Complete.\n")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    try:
        src = int(src)
    except (ValueError, TypeError):
        pass
    run(src)