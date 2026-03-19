"""
╔═══════════════════════════════════════════════════════════════╗
║                    AIR CANVAS  v5.0                          ║
║         Professional Gesture Drawing — No Mouse              ║
╠═══════════════════════════════════════════════════════════════╣
║  pip install mediapipe opencv-python numpy                   ║
║  python air_canvas.py                                        ║
╚═══════════════════════════════════════════════════════════════╝

GESTURES
  ☝  Index only           → DRAW / touch toolbar buttons
  ✌  Index + Middle (V)   → HOVER / pen-up / shape preview
  ✊  Fist (0 fingers)     → ERASER
  🖐  All 5 open (hold 2s) → CLEAR CANVAS

GESTURE SHORTCUTS
  👍 Thumb only           → BRUSH
  ✊ Fist                 → ERASER
  🤟 3 fingers           → NEXT COLOUR
  🖐 Open palm (hold)    → CLEAR CANVAS

KEYBOARD  (backup)
  S=save  C=clear  Z=undo  1-5=brush size  Q=quit
"""

import sys, urllib.request, pathlib, time, math, collections
import cv2, numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = pathlib.Path("hand_landmarker.task")
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")

def ensure_model():
    if MODEL_PATH.exists(): return
    print("[Downloading] hand_landmarker.task (~8 MB)...")
    def hook(b, bs, t):
        pct = min(b*bs,t)*100//max(t,1)
        done = "#" * (pct // 5)
        left = "-" * (20 - pct // 5)
        print(f"\r  {done}{left} {pct}%", end="", flush=True)
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=hook)
        print("\n[Ready]")
    except Exception as e:
        print(f"\n[Error] {e}"); sys.exit(1)

ensure_model()

import mediapipe as mp
from mediapipe.tasks.python        import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

# ─────────────────────────────────────────────────────────────────────────────
#  LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
W, H      = 1280, 720
PANEL_H   = 100
DRAW_H    = H - PANEL_H
WIN       = "Air Canvas  v5  |  Q=quit"

# ─────────────────────────────────────────────────────────────────────────────
#  THEME  — Professional Dark
# ─────────────────────────────────────────────────────────────────────────────
BG_PANEL   = ( 18,  18,  24)
BG_BLEND   = ( 10,  10,  14)
BORDER     = ( 48,  48,  72)
TEXT_DIM   = (100, 100, 130)
TEXT_BRIGHT= (220, 220, 240)
ACCENT     = ( 80, 160, 255)   # blue accent
ACCENT2    = (120, 220, 160)   # green accent
WARN       = ( 80,  80, 240)   # eraser / red

# Bone colours (neon green exo)
BONE_DEEP  = (  0,  30,  12)
BONE_MID   = (  0, 130,  55)
BONE_GLOW  = (  0, 220,  90)
JOINT_TIP  = (  0, 255, 160)
JOINT_NORM = ( 50, 200, 120)
PALM_ALPHA = 0.20

# ─────────────────────────────────────────────────────────────────────────────
#  PALETTE
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = [
    ("Red",    (  45,  45, 255)),
    ("Orange", (  20, 155, 255)),
    ("Yellow", (  20, 230, 220)),
    ("Lime",   (  50, 240,  80)),
    ("Cyan",   ( 220, 220,  20)),
    ("Blue",   ( 255, 110,  30)),
    ("Violet", ( 215,  40, 215)),
    ("Pink",   ( 185,  85, 255)),
    ("White",  ( 255, 255, 255)),
    ("Black",  (  15,  15,  15)),
]
CNAMES = [p[0] for p in PALETTE]
CBGRS  = [p[1] for p in PALETTE]

# ─────────────────────────────────────────────────────────────────────────────
#  TOOLS
# ─────────────────────────────────────────────────────────────────────────────
TOOLS   = ["pen", "eraser"]
TOOL_LBL= ["PEN", "ERASE"]
BRUSHES = [2, 4, 8, 14, 22, 32]

# Hand skeleton
BONES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
TIPS = [4, 8, 12, 16, 20]
PIPS = [3, 7, 11, 15, 19]
MCPS = [2, 6, 10, 14, 18]

# ─────────────────────────────────────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────────────────────────────────────
canvas      = np.zeros((H, W, 3), np.uint8)
undo_stack  = collections.deque(maxlen=30)

col_idx     = 0
tool_idx    = 0          # index into TOOLS
brush_sz    = 3          # actual pixel radius index into BRUSHES
prev_pt     = None
tip_hist    = collections.deque(maxlen=5)   # tip smoothing

# gesture stability — require N consecutive same gesture before switching
_gest_buf   = collections.deque(maxlen=6)
_cur_gest   = "none"
_last_raw_gest = "none"
_last_gesture_t = 0.0

# single-hand onboarding / lock
_scan_hand   = None
_scan_started= 0.0
_hand_locked = False

# toast notification
_toast_msg  = ""
_toast_end  = 0.0

# button touch cooldown
_last_touch = 0.0
TOUCH_CD    = 0.45

# gesture action cooldown / clear hold
_last_action_t = 0.0
_clear_since   = 0.0
_action_candidate = "none"
_action_since     = 0.0

# clear-canvas hold timer
# stroke tracking for undo
_stroke_pushed = False
_save_n        = 0

# ─────────────────────────────────────────────────────────────────────────────
#  BUTTON LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
BUTTONS = []

def _make_buttons():
    global BUTTONS
    BUTTONS = []
    # colour swatches — left 55% of panel
    nc    = len(PALETTE)
    zone  = int(W * 0.52)
    step  = zone // nc
    cy_sw = PANEL_H // 2 - 6
    for i in range(nc):
        cx = step//2 + i*step + 2
        BUTTONS.append({"k":"col","i":i,"cx":cx,"cy":cy_sw,"r":18})

    # tool buttons — middle strip
    nt   = len(TOOLS)
    tx0  = zone + 10
    tw   = 58
    tgap = 6
    for j in range(nt):
        x1 = tx0 + j*(tw+tgap)
        y1 = 12; y2 = PANEL_H-12
        BUTTONS.append({"k":"tool","i":j,
                        "x1":x1,"y1":y1,"x2":x1+tw,"y2":y2,
                        "cx":x1+tw//2,"cy":(y1+y2)//2})

    # brush +/− and undo on the right
    rx  = tx0 + nt*(tw+tgap) + 10
    aw  = 62
    for label, act in [("SAVE","save"),("UNDO","undo"),("B+","brush+"),("B-","brush-")]:
        y1=12; y2=PANEL_H-12
        BUTTONS.append({"k":"act","act":act,"lbl":label,
                        "x1":rx,"y1":y1,"x2":rx+aw,"y2":y2,
                        "cx":rx+aw//2,"cy":(y1+y2)//2})
        rx += aw + 6

_make_buttons()

# ─────────────────────────────────────────────────────────────────────────────
#  PANEL RENDER
# ─────────────────────────────────────────────────────────────────────────────
def render_panel(dst, hover_pt):
    p  = dst.copy()
    hx = hover_pt[0] if hover_pt else -9999
    hy = hover_pt[1] if hover_pt else -9999
    overlay = p.copy()

    cv2.line(overlay, (0, PANEL_H-1), (W, PANEL_H-1), (90, 90, 120), 1, cv2.LINE_AA)

    for btn in BUTTONS:
        k = btn["k"]

        if k == "col":
            i    = btn["i"]
            bgr  = CBGRS[i]
            sel  = (i == col_idx) and (TOOLS[tool_idx] != "eraser")
            hit  = math.hypot(hx-btn["cx"], hy-btn["cy"]) <= btn["r"]+10
            r    = btn["r"]
            cv2.circle(overlay, (btn["cx"], btn["cy"]), r+7, (0,0,0), -1, cv2.LINE_AA)
            cv2.circle(overlay, (btn["cx"], btn["cy"]), r, bgr, -1, cv2.LINE_AA)
            if hit:
                cv2.circle(overlay, (btn["cx"],btn["cy"]), r+6,
                           tuple(min(255,int(c*1.4)) for c in bgr), 2, cv2.LINE_AA)
            if sel:
                cv2.circle(overlay, (btn["cx"],btn["cy"]), r+4, bgr, 2, cv2.LINE_AA)
                cv2.circle(overlay, (btn["cx"],btn["cy"]), 4, (255,255,255), -1)
            cv2.putText(p, CNAMES[i][0],
                        (btn["cx"]-4, btn["cy"]+r+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, TEXT_DIM, 1, cv2.LINE_AA)

        elif k == "tool":
            j   = btn["i"]
            sel = (j == tool_idx)
            hit = btn["x1"]-8<=hx<=btn["x2"]+8 and btn["y1"]-8<=hy<=btn["y2"]+8
            x1,y1,x2,y2 = btn["x1"],btn["y1"],btn["x2"],btn["y2"]

            if sel:
                bg = tuple(int(c*0.38) for c in ACCENT)
                bd = ACCENT
            elif hit:
                bg = (34,38,54)
                bd = tuple(int(c*0.7) for c in ACCENT)
            else:
                bg = None
                bd = (78, 66, 66)

            _rrect(overlay, x1,y1,x2,y2, bg, bd, r=8)
            lbl = TOOL_LBL[j]
            tw2   = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0][0]
            cv2.putText(p, lbl, (btn["cx"]-tw2//2, btn["cy"]+6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        TEXT_BRIGHT if sel else TEXT_DIM, 1, cv2.LINE_AA)

        elif k == "act":
            hit = btn["x1"]-8<=hx<=btn["x2"]+8 and btn["y1"]-8<=hy<=btn["y2"]+8
            x1,y1,x2,y2 = btn["x1"],btn["y1"],btn["x2"],btn["y2"]
            bg = (24,44,34) if hit else None
            bd = ACCENT2 if hit else (54, 92, 60)
            _rrect(overlay, x1,y1,x2,y2, bg, bd, r=8)
            lbl = btn["lbl"]
            tw2 = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
            cv2.putText(p, lbl, (btn["cx"]-tw2//2, btn["cy"]+7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        ACCENT2 if hit else TEXT_DIM, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.55, p, 0.45, 0, p)

    # brush size indicator
    bsz = BRUSHES[brush_sz]
    cv2.putText(p, f"sz:{bsz}",
                (W-54, PANEL_H-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_DIM, 1, cv2.LINE_AA)

    cv2.putText(p,
        "Thumb=Brush  Fist=Eraser  3 Fingers=Color  Palm Hold=Clear",
        (6, PANEL_H-4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, TEXT_DIM, 1, cv2.LINE_AA)
    return p


def _rrect(img, x1,y1,x2,y2, fill, border, r=6):
    """Rounded rectangle helper."""
    if fill is not None:
        cv2.rectangle(img,(x1+r,y1),(x2-r,y2),fill,-1)
        cv2.rectangle(img,(x1,y1+r),(x2,y2-r),fill,-1)
        for cx2,cy2,a0 in [(x1+r,y1+r,180),(x2-r,y1+r,270),(x1+r,y2-r,90),(x2-r,y2-r,0)]:
            cv2.ellipse(img,(cx2,cy2),(r,r),0,a0,a0+90,fill,-1)
    cv2.rectangle(img,(x1,y1),(x2,y2),border,1)


# ─────────────────────────────────────────────────────────────────────────────
#  BUTTON HIT TEST
# ─────────────────────────────────────────────────────────────────────────────
def try_touch(px, py):
    global col_idx, tool_idx, brush_sz, canvas, _last_touch
    global prev_pt, _stroke_pushed
    now = time.time()
    if now - _last_touch < TOUCH_CD: return
    if py >= PANEL_H: return

    for btn in BUTTONS:
        k = btn["k"]
        if k == "col":
            if math.hypot(px-btn["cx"], py-btn["cy"]) <= btn["r"]+8:
                col_idx   = btn["i"]
                tool_idx  = 0   # switch to pen
                prev_pt = None
                _stroke_pushed = False
                _toast(f"● {CNAMES[col_idx]}", 1.0)
                _last_touch = now; return
        elif k == "tool":
            if btn["x1"]-6<=px<=btn["x2"]+6 and btn["y1"]-6<=py<=btn["y2"]+6:
                tool_idx = btn["i"]
                prev_pt = None
                _stroke_pushed = False
                _toast(f"Tool: {TOOLS[tool_idx].upper()}", 1.0)
                _last_touch = now; return
        elif k == "act":
            if btn["x1"]-6<=px<=btn["x2"]+6 and btn["y1"]-6<=py<=btn["y2"]+6:
                if btn["act"] == "brush+":
                    brush_sz = min(brush_sz+1, len(BRUSHES)-1)
                    _toast(f"Size: {BRUSHES[brush_sz]}px", 0.8)
                elif btn["act"] == "brush-":
                    brush_sz = max(brush_sz-1, 0)
                    _toast(f"Size: {BRUSHES[brush_sz]}px", 0.8)
                elif btn["act"] == "undo":
                    _do_undo()
                elif btn["act"] == "save":
                    _save_canvas()
                _last_touch = now; return

def _do_undo():
    if undo_stack:
        canvas[:] = undo_stack.pop()
        _toast("Undo ✓", 1.0)
    else:
        _toast("Nothing to undo", 1.0)

def _push_undo():
    undo_stack.append(canvas.copy())

def _toast(msg, dur=1.2):
    global _toast_msg, _toast_end
    _toast_msg = msg
    _toast_end = time.time() + dur


def _save_canvas():
    global _save_n
    _save_n += 1
    fn = f"air_canvas_{_save_n:03d}.png"
    cv2.imwrite(fn, canvas)
    _toast(f"Saved {fn}", 1.2)
    print(f"[Saved] {fn}")


# ─────────────────────────────────────────────────────────────────────────────
#  FINGER COUNT  — robust 3-D distance method
# ─────────────────────────────────────────────────────────────────────────────
def count_fingers(lm):
    """
    Each finger counted extended only when:
      • tip distance from wrist > MCP distance from wrist × 1.10  (stretched out)
      • tip.y < pip.y - threshold  (tip above the middle knuckle)
    Thumb uses thumb-tip vs index-MCP lateral distance.
    Returns 0-5 with high accuracy across orientations.
    """
    wrist = np.array([lm[0].x, lm[0].y, lm[0].z])
    count = 0

    # Thumb
    t_tip = np.array([lm[4].x, lm[4].y, lm[4].z])
    t_ip  = np.array([lm[3].x, lm[3].y, lm[3].z])
    i_mcp = np.array([lm[5].x, lm[5].y, lm[5].z])
    if np.linalg.norm(t_tip - i_mcp) > np.linalg.norm(t_ip - i_mcp) * 1.05:
        count += 1

    # Fingers 2-5
    for tip_i, pip_i, mcp_i in zip(TIPS[1:], PIPS[1:], MCPS[1:]):
        tip = np.array([lm[tip_i].x, lm[tip_i].y, lm[tip_i].z])
        pip = np.array([lm[pip_i].x, lm[pip_i].y, lm[pip_i].z])
        mcp = np.array([lm[mcp_i].x, lm[mcp_i].y, lm[mcp_i].z])

        far_enough = np.linalg.norm(tip - wrist) > np.linalg.norm(mcp - wrist) * 1.10
        tip_up     = lm[tip_i].y < lm[pip_i].y - 0.012

        if far_enough and tip_up:
            count += 1

    return count


def stable_gesture(raw):
    """Only return a new gesture after it appears N times in a row."""
    global _cur_gest
    _gest_buf.append(raw)
    # if last 4 are all same, commit
    if len(_gest_buf) >= 4 and len(set(list(_gest_buf)[-4:])) == 1:
        _cur_gest = raw
    return _cur_gest


def classify_gesture(fc):
    """Map finger count → gesture name."""
    if   fc == 0: return "erase"
    elif fc == 1: return "draw"
    elif fc == 2: return "hover"
    elif fc == 5: return "clear"
    else:         return "idle"


# ─────────────────────────────────────────────────────────────────────────────
#  PINCH  (thumb-index distance → brush size)
# ─────────────────────────────────────────────────────────────────────────────
def _lm_xyz(lm, idx):
    return np.array([lm[idx].x, lm[idx].y, lm[idx].z], dtype=np.float32)


def _dist3(a, b):
    return float(np.linalg.norm(a - b))


def _finger_states(lm):
    wrist = _lm_xyz(lm, 0)
    states = {}

    thumb_tip = _lm_xyz(lm, 4)
    thumb_ip = _lm_xyz(lm, 3)
    thumb_mcp = _lm_xyz(lm, 2)
    index_mcp = _lm_xyz(lm, 5)
    thumb_reach = _dist3(thumb_tip, index_mcp) / max(_dist3(thumb_mcp, index_mcp), 1e-6)
    thumb_straight = _dist3(thumb_tip, thumb_mcp) > _dist3(thumb_ip, thumb_mcp) * 1.18
    states["thumb"] = thumb_reach > 1.10 and thumb_straight

    for name, tip_i, pip_i, mcp_i in [
        ("index", 8, 7, 5),
        ("middle", 12, 11, 9),
        ("ring", 16, 15, 13),
        ("pinky", 20, 19, 17),
    ]:
        tip = _lm_xyz(lm, tip_i)
        pip = _lm_xyz(lm, pip_i)
        mcp = _lm_xyz(lm, mcp_i)
        reach = _dist3(tip, wrist) / max(_dist3(mcp, wrist), 1e-6)
        straight = _dist3(tip, mcp) > _dist3(pip, mcp) * 1.18
        tip_up = lm[tip_i].y < lm[pip_i].y - 0.010
        states[name] = reach > 1.12 and straight and tip_up

    return states


def stable_gesture(raw, now):
    """Gesture hysteresis with a short grace period to avoid mid-stroke dropouts."""
    global _cur_gest, _last_raw_gest, _last_gesture_t
    if raw != _last_raw_gest:
        _gest_buf.clear()
        _last_raw_gest = raw
    _gest_buf.append(raw)

    needed = 2 if raw in ("idle", "none") else 3
    if len(_gest_buf) >= needed and len(set(_gest_buf)) == 1:
        _cur_gest = raw
        if raw in ("draw", "hover"):
            _last_gesture_t = now
    elif raw in ("idle", "none") and _cur_gest in ("draw", "hover") and now - _last_gesture_t < 0.14:
        return _cur_gest
    return _cur_gest


def classify_gesture(lm):
    fingers = _finger_states(lm)
    fc = sum(fingers.values())

    if all(fingers.values()):
        return fc, "clear"

    if fingers["thumb"] and not fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
        return fc, "brush"

    if not any(fingers.values()):
        return fc, "erase"

    if fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
        return fc, "draw"

    if fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
        return fc, "hover"

    if fingers["index"] and fingers["middle"] and fingers["ring"] and not fingers["pinky"]:
        return fc, "color"

    return fc, "idle"


def extract_handedness(result):
    try:
        if not result.handedness:
            return None
        entry = result.handedness[0]
        if not entry:
            return None
        item = entry[0]
        return getattr(item, "category_name", None) or getattr(item, "display_name", None)
    except Exception:
        return None


def draw_scan_overlay(frame, hand_label, progress):
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (W, H), (8, 10, 16), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    card_x1, card_y1 = W//2 - 280, H//2 - 150
    card_x2, card_y2 = W//2 + 280, H//2 + 150
    card = frame.copy()
    cv2.rectangle(card, (card_x1, card_y1), (card_x2, card_y2), (20, 24, 34), -1)
    cv2.rectangle(card, (card_x1, card_y1), (card_x2, card_y2), (72, 84, 110), 1)
    cv2.addWeighted(card, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y1 + 6), ACCENT, -1)

    title = "Air Canvas"
    sub = "Professional gesture sketching"
    hand = f"Detected hand: {hand_label or 'waiting...'}"
    tip0 = "Raise one hand, hold still, then use that hand only"
    tip1 = "Index = draw"
    tip2 = "Index + middle = hover / toolbar"
    tip3 = "Thumb = brush   Fist = eraser"
    tip4 = "3 fingers = next colour   Open palm hold = clear all"

    cv2.putText(frame, title, (card_x1 + 36, card_y1 + 54),
                cv2.FONT_HERSHEY_SIMPLEX, 1.25, TEXT_BRIGHT, 2, cv2.LINE_AA)
    cv2.putText(frame, sub, (card_x1 + 36, card_y1 + 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (190, 190, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, hand, (card_x1 + 36, card_y1 + 138),
                cv2.FONT_HERSHEY_SIMPLEX, 0.66, ACCENT2 if hand_label else TEXT_DIM, 2, cv2.LINE_AA)
    cv2.putText(frame, tip0, (card_x1 + 36, card_y1 + 176),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(frame, tip1, (card_x1 + 36, card_y1 + 206),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, TEXT_BRIGHT, 1, cv2.LINE_AA)
    cv2.putText(frame, tip2, (card_x1 + 36, card_y1 + 234),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, TEXT_BRIGHT, 1, cv2.LINE_AA)
    cv2.putText(frame, tip3, (card_x1 + 36, card_y1 + 262),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, TEXT_BRIGHT, 1, cv2.LINE_AA)
    cv2.putText(frame, tip4, (card_x1 + 36, card_y1 + 290),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, TEXT_BRIGHT, 1, cv2.LINE_AA)

    x1, y1, x2, y2 = card_x1 + 36, card_y2 - 30, card_x2 - 36, card_y2 - 12
    cv2.rectangle(frame, (x1, y1), (x2, y2), BORDER, 1)
    cv2.rectangle(frame, (x1 + 2, y1 + 2), (x1 + 2 + int((x2 - x1 - 4) * progress), y2 - 2), ACCENT, -1)


# ─────────────────────────────────────────────────────────────────────────────
#  SKELETON RENDERER
# ─────────────────────────────────────────────────────────────────────────────
def _z_col(base, z):
    """Depth-modulated colour; z ~[-0.1, 0.1], closer=brighter."""
    t = max(0.0, min(1.0, 0.5 - z * 6.0))
    return tuple(int(c*(0.30 + 0.70*t)) for c in base)

def draw_skeleton(frame, lm, fw, fh, gesture, tip_scr):
    px = [(int(l.x*fw), int(l.y*fh)) for l in lm]

    # ── Palm fill ───────────────────────────────────────────────────────────
    palm_ids = [0,1,5,9,13,17]
    poly     = np.array([px[i] for i in palm_ids], np.int32)
    ov       = frame.copy()
    cv2.fillConvexPoly(ov, poly, BONE_DEEP)
    cv2.addWeighted(ov, PALM_ALPHA, frame, 1-PALM_ALPHA, 0, frame)
    cv2.polylines(frame, [poly], True, BONE_MID, 1, cv2.LINE_AA)

    # ── Bones ────────────────────────────────────────────────────────────────
    for a, b in BONES:
        z   = (lm[a].z + lm[b].z) * 0.5
        # 3-layer glow: deep core → mid → bright edge
        cv2.line(frame, px[a], px[b], _z_col(BONE_DEEP, z),  9, cv2.LINE_AA)
        cv2.line(frame, px[a], px[b], _z_col(BONE_MID,  z),  4, cv2.LINE_AA)
        cv2.line(frame, px[a], px[b], _z_col(BONE_GLOW, z),  2, cv2.LINE_AA)

    # ── Joints ───────────────────────────────────────────────────────────────
    for i, p in enumerate(px):
        z   = lm[i].z
        tip = i in TIPS
        r   = 9 if tip else 5
        cv2.circle(frame, p, r+5, _z_col(BONE_DEEP, z), -1)
        cv2.circle(frame, p, r+2, _z_col(BONE_MID,  z), -1)
        cv2.circle(frame, p, r,   _z_col(JOINT_TIP if tip else JOINT_NORM, z), -1)
        if tip:  # bright centre dot on fingertips
            cv2.circle(frame, p, 3, (200,255,220), -1)

    # Wrist
    cv2.circle(frame, px[0], 15, _z_col(BONE_DEEP, lm[0].z), -1)
    cv2.circle(frame, px[0], 15, _z_col(BONE_MID,  lm[0].z),  2)
    cv2.circle(frame, px[0],  6, _z_col(JOINT_NORM, lm[0].z), -1)

    # ── Gesture badge ────────────────────────────────────────────────────────
    GBADGE = {
        "draw":  ("☝ DRAW",  ACCENT2),
        "hover": ("✌ HOVER", (200,200,80)),
        "brush": ("BRUSH", ACCENT),
        "erase": ("ERASE", (160,120,255)),
        "color": ("COLOR", (255,200,80)),
        "clear": ("CLEAR", (120,220,200)),
        "idle":  ("✋ IDLE",  TEXT_DIM),
    }
    if gesture in GBADGE:
        lbl, col = GBADGE[gesture]
        bx, by   = px[0][0]-50, px[0][1]+45
        cv2.rectangle(frame,(bx-4,by-20),(bx+160,by+8),(12,12,18),1)
        cv2.putText(frame, lbl, (bx, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, col, 2, cv2.LINE_AA)

    # ── Cursor at index tip ──────────────────────────────────────────────────
    if tip_scr:
        tx, ty = tip_scr
        if ty > PANEL_H:
            tool = TOOLS[tool_idx]
            if tool == "eraser":
                cc = WARN;   cr = 50
            else:
                cc = CBGRS[col_idx]
                cr = BRUSHES[brush_sz] + 4

            for dr, alpha in [(10,0.05),(6,0.12),(0,1.0)]:
                r_col = tuple(int(c*alpha) for c in cc)
                cv2.circle(frame, (tx,ty), cr+dr, r_col,
                           -1 if dr > 0 else 2, cv2.LINE_AA)
            cv2.circle(frame, (tx,ty), 3, (255,255,255), -1)


# ─────────────────────────────────────────────────────────────────────────────
#  SHAPE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
#  HUD
# ─────────────────────────────────────────────────────────────────────────────
def draw_hud(blend, fc, gesture, now):
    if now < _toast_end:
        msg = _toast_msg
        col = (240,240,80)
    elif TOOLS[tool_idx] == "eraser":
        msg = "  ✊  ERASER"
        col = (140,110,255)
    else:
        sz  = BRUSHES[brush_sz]
        msg = f"  ● {CNAMES[col_idx]}   PEN   {sz}px  |  gestures active"
        col = TEXT_BRIGHT

    cv2.putText(blend, msg, (12,34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.64, (0,0,0), 5, cv2.LINE_AA)
    cv2.putText(blend, msg, (10,32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.64, col, 2, cv2.LINE_AA)

    uc = len(undo_stack)
    cv2.putText(blend, f"undo:{uc}", (W-92,34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(blend, f"undo:{uc}", (W-90,32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_DIM, 1, cv2.LINE_AA)

    if _hand_locked and _scan_hand:
        tag = f"{_scan_hand.upper()} HAND LOCKED"
        cv2.putText(blend, tag, (10, H-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(blend, tag, (10, H-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, ACCENT2, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global prev_pt, brush_sz, col_idx, tool_idx, canvas
    global _stroke_pushed
    global _cur_gest, _last_raw_gest, _last_gesture_t
    global _scan_hand, _scan_started, _hand_locked
    global _last_action_t, _clear_since
    global _action_candidate, _action_since

    opts = HandLandmarkerOptions(
        base_options  = BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode  = RunningMode.VIDEO,
        num_hands     = 1,
        min_hand_detection_confidence = 0.62,
        min_hand_presence_confidence  = 0.62,
        min_tracking_confidence       = 0.58,
    )
    det = HandLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)

    ts = 0
    print("\n  Air Canvas v5 ready - show your hand!\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        ts += 33
        now = time.time()

        # ── detect ────────────────────────────────────────────────────────────
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = det.detect_for_video(mp_img, ts)

        gesture = "none"; tip_scr = None; fc = 0; hand_label = None

        if result.hand_landmarks:
            hand_label = extract_handedness(result)
            if not _hand_locked:
                if hand_label:
                    if _scan_hand != hand_label:
                        _scan_hand = hand_label
                        _scan_started = now
                    elif now - _scan_started >= 0.85:
                        _hand_locked = True
                        _toast(f"{_scan_hand.upper()} hand locked", 1.2)
                else:
                    _scan_started = 0.0

            if _hand_locked and hand_label and hand_label != _scan_hand:
                tip_hist.clear()
            else:
                lm  = result.hand_landmarks[0]
                fc, raw = classify_gesture(lm)
                gesture = stable_gesture(raw, now)

            # smoothed tip
                ix = int(lm[8].x * fw); iy = int(lm[8].y * fh)
                tip_hist.append((ix, iy))
                tip_scr = (
                    int(sum(p[0] for p in tip_hist) / len(tip_hist)),
                    int(sum(p[1] for p in tip_hist) / len(tip_hist)),
                )

            # draw exoskeleton
                draw_skeleton(frame, lm, fw, fh, gesture, tip_scr)

            # ── gesture actions ───────────────────────────────────────────────
            if gesture == "clear":
                if _clear_since == 0.0:
                    _clear_since = now
                elif now - _clear_since >= 1.2:
                    _push_undo()
                    canvas[:] = 0
                    prev_pt = None
                    _stroke_pushed = False
                    _clear_since = 0.0
                    _toast("Canvas cleared", 1.0)
            else:
                _clear_since = 0.0

            if gesture in ("brush", "erase", "color"):
                if _action_candidate != gesture:
                    _action_candidate = gesture
                    _action_since = now
                elif now - _action_since >= 0.25 and now - _last_action_t >= 2.0:
                    if gesture == "brush":
                        tool_idx = 0
                        _toast("Tool: PEN", 0.8)
                    elif gesture == "erase":
                        tool_idx = 1
                        _toast("Tool: ERASER", 0.8)
                    else:
                        col_idx = (col_idx + 1) % len(CBGRS)
                        tool_idx = 0
                        _toast(f"● {CNAMES[col_idx]}", 0.8)
                    prev_pt = None
                    _stroke_pushed = False
                    _last_action_t = now
                    _action_candidate = "none"
                    _action_since = 0.0
            else:
                _action_candidate = "none"
                _action_since = 0.0

            # ── clear (hold open hand) ────────────────────────────────────────

            # ── toolbar touch ─────────────────────────────────────────────────
            if not _hand_locked:
                draw_scan_overlay(frame, hand_label, min(max((now - _scan_started) / 0.85, 0.0), 1.0) if _scan_started else 0.0)
                prev_pt = None; _stroke_pushed = False
            elif tip_scr and gesture == "hover" and tip_scr[1] < PANEL_H:
                try_touch(tip_scr[0], tip_scr[1])
                prev_pt = None; _stroke_pushed = False

            # ── drawing area ──────────────────────────────────────────────────
            elif tip_scr:
                tool = TOOLS[tool_idx]
                tp   = tip_scr   # full-display coords
                tp_canvas = (tp[0], tp[1])  # canvas has same size

                if tool == "eraser":
                    if gesture in ("draw", "erase"):
                        if not _stroke_pushed:
                            _push_undo(); _stroke_pushed = True
                        cv2.circle(canvas, tp_canvas, 50, (0,0,0), -1)
                    else:
                        _stroke_pushed = False
                    prev_pt = None

                elif tool == "pen" and gesture == "draw":
                    if not _stroke_pushed:
                        _push_undo(); _stroke_pushed = True
                    if prev_pt:
                        cv2.line(canvas, prev_pt, tp_canvas,
                                 CBGRS[col_idx], BRUSHES[brush_sz], cv2.LINE_AA)
                    prev_pt = tp_canvas

                else:
                    prev_pt = None; _stroke_pushed = False
            else:
                prev_pt = None; _stroke_pushed = False

        else:
            tip_hist.clear(); prev_pt = None
            _stroke_pushed = False
            _action_candidate = "none"
            _action_since = 0.0
            if not _hand_locked or now - _last_gesture_t > 0.16:
                _gest_buf.clear()
                _cur_gest = "none"
                _last_raw_gest = "none"
                gesture = "none"

        # ── composite ─────────────────────────────────────────────────────────
        cam  = cv2.resize(frame, (W, H))
        ink  = canvas
        gray = cv2.cvtColor(ink, cv2.COLOR_BGR2GRAY)
        _, mk= cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
        mk3  = cv2.merge([mk,mk,mk])
        blend= cv2.add(cv2.bitwise_and(cam, cv2.bitwise_not(mk3)),
                       cv2.bitwise_and(ink, mk3))

        if _clear_since > 0.0:
            prog = min((now - _clear_since) / 1.2, 1.0)
            cv2.rectangle(blend, (0, H-6), (int(W * prog), H), WARN, -1)
            cv2.putText(blend, "Hold palm to clear all", (12, H-18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, WARN, 1, cv2.LINE_AA)

        if not _hand_locked:
            prog = min(max((now - _scan_started) / 0.85, 0.0), 1.0) if _scan_started else 0.0
            draw_scan_overlay(blend, hand_label, prog)

        draw_hud(blend, fc, gesture, now)

        hover_panel = tip_scr if (tip_scr and gesture == "hover" and tip_scr[1] < PANEL_H) else None
        output = render_panel(blend, hover_panel)
        cv2.imshow(WIN, output)

        # ── keyboard ──────────────────────────────────────────────────────────
        k = cv2.waitKey(1) & 0xFF
        if   k == ord('q'): break
        elif k == ord('c'): _push_undo(); canvas[:]=0; _toast("Cleared!")
        elif k == ord('z'): _do_undo()
        elif k == ord('s'):
            _save_canvas()
        elif ord('1')<=k<=ord('5'): brush_sz=k-ord('1')

    cap.release(); cv2.destroyAllWindows(); det.close()

if __name__ == "__main__":
    main()
