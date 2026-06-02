"""
╔══════════════════════════════════════════════════════════════════╗
║         AI FLOOD MONITORING & RISK PREDICTION SYSTEM            ║
║                  Production-Grade Computer Vision                ║
╠══════════════════════════════════════════════════════════════════╣
║  Dev/Creator : tubakhxn                                          ║
║  GitHub      : https://github.com/tubakhxn                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import subprocess, sys, os, time, importlib

OUTPUT_FILE = "output_flood_monitoring_ai.mp4"

PACKAGES = [
    "ultralytics", "opencv-python", "numpy", "torch", "torchvision",
    "scipy", "pillow", "matplotlib", "supervision", "mediapipe"
]

def install_packages():
    print("\n┌─────────────────────────────────────────────┐")
    print("│  AI FLOOD MONITORING — DEPENDENCY INSTALLER  │")
    print("└─────────────────────────────────────────────┘")
    for pkg in PACKAGES:
        mod = pkg.replace("-", "_").split("==")[0]
        if mod == "opencv_python": mod = "cv2"
        if mod == "pillow": mod = "PIL"
        try:
            importlib.import_module(mod)
            print(f"  ✓  {pkg}")
        except ImportError:
            print(f"  ↓  Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
            print(f"  ✓  {pkg}")

install_packages()

import cv2, numpy as np, torch, math
from datetime import datetime
from collections import deque

try:
    from ultralytics import YOLO
    YOLO_OK = True
except: YOLO_OK = False

# ─── DEVICE SETUP ────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = DEVICE == "cuda"
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
print(f"\n    Device : {GPU_NAME}")
print(f"   FP16   : {USE_FP16}\n")

# ─── COLOUR PALETTE ──────────────────────────────────────────────
C = {
    "bg":       (10, 14, 20),
    "panel":    (15, 22, 35),
    "border":   (0, 120, 200),
    "water":    (30, 144, 255),
    "water2":   (0, 80, 180),
    "safe":     (0, 220, 120),
    "warn":     (0, 200, 255),
    "danger":   (0, 100, 255),
    "critical": (0, 0, 230),
    "text":     (220, 235, 255),
    "accent":   (100, 200, 255),
    "glow":     (60, 160, 255),
}

def hex_to_bgr(h):
    h = h.lstrip("#")
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return (b,g,r)

def draw_text_shadow(img, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX,
                     scale=0.5, color=(255,255,255), thick=1):
    x,y = pos
    cv2.putText(img, text, (x+1,y+1), font, scale, (0,0,0), thick+1, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thick, cv2.LINE_AA)

def draw_glowing_rect(img, pt1, pt2, color, thick=2, glow_alpha=0.3):
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, glow_alpha, img, 1-glow_alpha, 0, img)
    cv2.rectangle(img, pt1, pt2, color, thick)
    gc = tuple(min(255, int(c*1.5)) for c in color)
    cv2.rectangle(img, (pt1[0]-1, pt1[1]-1), (pt2[0]+1, pt2[1]+1), gc, 1)

def estimate_water_mask(frame):
    """Segment water-like regions using HSV colour + edge cues."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Blue-water range
    m1 = cv2.inRange(hsv, (90,30,30), (140,255,255))
    # Dark-water / reflective
    m2 = cv2.inRange(hsv, (85,10,0), (145,80,120))
    # Green-murky water
    m3 = cv2.inRange(hsv, (35,30,30), (90,200,180))
    mask = cv2.bitwise_or(m1, cv2.bitwise_or(m2, m3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((15,15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            np.ones((8,8), np.uint8))
    mask = cv2.GaussianBlur(mask, (7,7), 0)
    _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
    return mask

def classify_risk(coverage_pct, bottom_thirds_pct):
    if coverage_pct > 60 or bottom_thirds_pct > 75:
        return "CRITICAL", C["critical"]
    if coverage_pct > 35 or bottom_thirds_pct > 50:
        return "DANGER", C["danger"]
    if coverage_pct > 15 or bottom_thirds_pct > 25:
        return "WARNING", C["warn"]
    return "SAFE", C["safe"]

def draw_hud(frame, stats, frame_idx, total_frames, fps_proc, start_time):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # ── Top banner ──────────────────────────────────────────────
    cv2.rectangle(overlay, (0,0), (w, 56), C["panel"], -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (0,56), (w,56), C["border"], 1)
    draw_text_shadow(frame, "◈  AI FLOOD MONITORING & RISK PREDICTION SYSTEM",
                     (14, 22), scale=0.65, color=C["accent"], thick=1)
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    draw_text_shadow(frame, ts, (w-220, 22), scale=0.45, color=C["text"])
    draw_text_shadow(frame, f"DEV: tubakhxn  |  github.com/tubakhxn",
                     (14, 46), scale=0.38, color=C["glow"])

    # ── Left status panel ───────────────────────────────────────
    px, py, pw, ph = 12, 68, 220, 310
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (px,py), (px+pw, py+ph), C["panel"], -1)
    cv2.addWeighted(overlay2, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (px,py), (px+pw, py+ph), C["border"], 1)
    draw_text_shadow(frame, "[ FLOOD INTELLIGENCE ]", (px+8, py+18),
                     scale=0.42, color=C["accent"])
    cv2.line(frame, (px+6, py+26), (px+pw-6, py+26), C["border"], 1)

    rows = [
        ("COVERAGE",    f"{stats['coverage']:.1f}%"),
        ("WATER AREA",  f"{stats['water_px']:,} px"),
        ("DEPTH EST.",  f"{stats['depth_est']:.1f} m"),
        ("RISK SCORE",  f"{stats['risk_score']:.0f}/100"),
        ("ACTIVE ZONES",f"{stats['zones']}"),
        ("INFRAST. HIT",f"{stats['infra_hits']}"),
    ]
    for i,(k,v) in enumerate(rows):
        y = py + 46 + i*38
        draw_text_shadow(frame, k, (px+10, y), scale=0.36, color=C["glow"])
        draw_text_shadow(frame, v, (px+10, y+17), scale=0.48, color=C["text"], thick=1)

    # ── Risk badge ──────────────────────────────────────────────
    level, lcol = stats["level"], stats["lcol"]
    bx, by = px+14, py+ph-48
    draw_glowing_rect(frame, (bx, by), (bx+192, by+36), lcol, thick=2, glow_alpha=0.35)
    draw_text_shadow(frame, f"● {level}", (bx+12, by+24),
                     scale=0.65, color=lcol, thick=2)

    # ── Bottom progress bar ─────────────────────────────────────
    bar_y = h - 36
    cv2.rectangle(frame, (0, bar_y-2), (w, h), C["panel"], -1)
    cv2.line(frame, (0, bar_y-2), (w, bar_y-2), C["border"], 1)
    prog = frame_idx / max(total_frames, 1)
    bw = int(w * prog)
    cv2.rectangle(frame, (0, bar_y+4), (bw, bar_y+14), C["border"], -1)
    cv2.rectangle(frame, (0, bar_y+4), (w, bar_y+14), C["glow"], 1)
    elapsed = time.time() - start_time
    eta = (elapsed / max(prog, 0.001)) * (1-prog)
    draw_text_shadow(frame, f"PROGRESS: {prog*100:.1f}%", (10, bar_y+28),
                     scale=0.42, color=C["text"])
    draw_text_shadow(frame, f"ETA: {int(eta)}s", (200, bar_y+28),
                     scale=0.42, color=C["text"])
    draw_text_shadow(frame, f"FPS: {fps_proc:.1f}", (340, bar_y+28),
                     scale=0.42, color=C["accent"])
    draw_text_shadow(frame, f"FRAME: {frame_idx}/{total_frames}", (440, bar_y+28),
                     scale=0.42, color=C["text"])
    dev_str = f"GPU: {GPU_NAME[:22]}"
    draw_text_shadow(frame, dev_str, (w-260, bar_y+28),
                     scale=0.38, color=C["glow"])

    # ── Scanlines (light) ───────────────────────────────────────
    for y in range(0, h, 4):
        cv2.line(frame, (0,y), (w,y), (0,0,0), 1)
        # blend very lightly
    return frame

def process_video(src):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"  ✗  Cannot open source: {src}")
        return

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 9999

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps_in, (fw, fh))

    # Accumulation map
    accum_map = np.zeros((fh, fw), np.float32)

    model = None
    if YOLO_OK:
        try:
            model = YOLO("yolov8n.pt")
            if USE_FP16: model.model.half()
            print("  ✓  YOLOv8 loaded")
        except: pass

    print(f"\n  ▶  Processing  {total_frames} frames …\n")
    start_time = time.time()
    frame_idx = 0
    fps_proc = 0.0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        # ── Water segmentation ──────────────────────────────────
        water_mask = estimate_water_mask(frame)
        water_px = int(np.sum(water_mask > 0))
        total_px = fw * fh
        coverage = water_px / total_px * 100

        # lower third coverage
        lower_mask = water_mask[int(fh*0.67):, :]
        lower_cov = np.sum(lower_mask > 0) / (fw * int(fh*0.33) + 1) * 100

        depth_est = coverage / 20.0 * 2.5          # rough pseudo-depth
        risk_score = min(100, coverage * 1.2 + lower_cov * 0.8)
        level, lcol = classify_risk(coverage, lower_cov)

        # Update accumulation map
        accum_map += (water_mask.astype(np.float32) / 255.0) * 0.05
        accum_map = np.clip(accum_map, 0, 1)

        # ── Overlay water coloring ──────────────────────────────
        vis = frame.copy()
        water_col_overlay = np.zeros_like(frame)
        water_col_overlay[water_mask > 0] = C["water"]
        cv2.addWeighted(water_col_overlay, 0.35, vis, 0.65, 0, vis)

        # Accumulation heatmap
        accum_vis = cv2.applyColorMap(
            (accum_map * 255).astype(np.uint8), cv2.COLORMAP_OCEAN)
        cv2.addWeighted(accum_vis, 0.18, vis, 0.82, 0, vis)

        # Contour outlines
        contours, _ = cv2.findContours(
            water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, C["water"], 2)

        # YOLO vehicle / road detection
        infra_hits = 0
        zones = len(contours)
        if model and frame_idx % 3 == 0:
            results = model(frame, verbose=False,
                            half=USE_FP16, device=DEVICE)[0]
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < 0.35: continue
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                label = model.names[cls]
                # Check if box is in water zone
                cx, cy = (x1+x2)//2, (y1+y2)//2
                if 0<=cy<fh and 0<=cx<fw and water_mask[cy,cx] > 0:
                    infra_hits += 1
                    draw_glowing_rect(vis, (x1,y1),(x2,y2), C["critical"], thick=2)
                    draw_text_shadow(vis, f"⚠ {label.upper()}", (x1, y1-6),
                                     scale=0.45, color=C["critical"], thick=1)
                else:
                    cv2.rectangle(vis, (x1,y1),(x2,y2), C["safe"], 1)

        stats = dict(coverage=coverage, water_px=water_px,
                     depth_est=depth_est, risk_score=risk_score,
                     zones=zones, infra_hits=infra_hits,
                     level=level, lcol=lcol)

        # ── FPS calc ────────────────────────────────────────────
        fps_proc = frame_idx / max(time.time()-start_time, 0.001)

        vis = draw_hud(vis, stats, frame_idx, total_frames, fps_proc, start_time)
        out.write(vis)

        # Console progress
        if frame_idx % 30 == 0:
            prog = frame_idx/total_frames*100
            elapsed = time.time()-start_time
            eta = elapsed/max(prog/100,0.001)*(1-prog/100)
            bar = "█"*int(prog//4) + "░"*(25-int(prog//4))
            print(f"\r  [{bar}] {prog:5.1f}%  FPS:{fps_proc:5.1f}  ETA:{int(eta)}s  "
                  f"Level:{level}      ", end="", flush=True)

    cap.release()
    out.release()
    print(f"\n\n    Done!  →  {OUTPUT_FILE}\n")

# ─── ENTRY POINT ─────────────────────────────────────────────────
if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    print(f"""
╔══════════════════════════════════════════════════════════╗
║     AI FLOOD MONITORING & RISK PREDICTION SYSTEM        ║
║             tubakhxn  |  github.com/tubakhxn            ║
╚══════════════════════════════════════════════════════════╝
  Source  : {src}
  Device  : {GPU_NAME}
  Output  : {OUTPUT_FILE}
""")
    process_video(src)