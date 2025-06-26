"""
student_affect_monitor.py              (dlib + DeepFace version)
────────────────────────────────────────────────────────────────
Everything is identical to your original script except:
• MediaPipe face-mesh replaced by dlib 68-landmark detector
• EAR tiredness computed from those landmarks
"""

import cv2, threading, time, statistics, sys, platform, os
from collections import Counter
from deepface import DeepFace
import dlib 
from imutils import face_utils          # pip install imutils
from pynput import keyboard, mouse
from scipy.spatial import distance as dist
import tkinter as tk

# ───────── Tunables (unchanged) ─────────
EMOTION_SAMPLE_INTERVAL  = 1
EMOTION_REPORT_INTERVAL  = 30
ALERT_THRESHOLD          = 5
EAR_THRESH               = 0.23
EAR_CONSEC_FRAMES        = 15

# ───────── Runtime globals ──────────────
emotion_window, typing_metrics = [], []
scroll_events, eye_closed_counter = 0, 0
key_press_times = {}

# ───────── dlib face-landmark setup ─────
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"  # ← put model here
detector   = dlib.get_frontal_face_detector()
predictor  = dlib.shape_predictor(PREDICTOR_PATH)

# dlib indices (68-point) for each eye
LEFT_EYE  = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

# ───────── Mapping helper (unchanged) ──
RAW_TO_HIGH = {
    "angry"   : "tension/frustration",
    "fear"    : "tension/frustration",
    "disgust" : "confusion",
    "surprise": "confusion",
    "sad"     : "boredom",
    "happy"   : "engagement/focus",
    "neutral" : "engagement/focus"
}
NEGATIVE_EMOTIONS = {"tired","tension/frustration","confusion","boredom"}

def harmonise(raw_face, tired_flag):
    return "tired" if tired_flag else RAW_TO_HIGH.get(raw_face, "engagement/focus")

# ───────── EAR utility (unchanged) ─────
def ear(pts):
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)

# ───────── Alert helpers (unchanged) ───
def play_sound():
    try:
        if platform.system() == "Windows":
            import winsound
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        elif platform.system() == "Darwin":
            os.system('afplay /System/Library/Sounds/Pop.aiff')
        else:
            print('\a', end='', flush=True)
    except Exception:
        pass

def show_popup(emotion):
    play_sound()
    msg = f"⚠  Frequent {emotion.upper()} detected!"
    if platform.system() == "Windows":
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, msg, "Emotion Alert", 0x30)
    else:
        from tkinter import messagebox
        root = tk.Tk(); root.withdraw()
        messagebox.showwarning("Emotion Alert", msg)
        root.destroy()

# ───────── Webcam thread (MediaPipe → dlib) ─────────
def detect_emotion():
    global eye_closed_counter
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform.startswith("win") else 0)
    if not cam.isOpened():
        sys.stderr.write("❌  Cannot access webcam.\n"); return

    last_inf = 0.0
    while True:
        ok, frame = cam.read()
        if not ok: continue
        now, tired_now = time.time(), False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for rect in faces:            # only first face is used
            shape = predictor(gray, rect)
            shape_np = face_utils.shape_to_np(shape)
            l_eye = shape_np[LEFT_EYE]
            r_eye = shape_np[RIGHT_EYE]
            if ear(l_eye) < EAR_THRESH and ear(r_eye) < EAR_THRESH:
                eye_closed_counter += 1
            else:
                if eye_closed_counter >= EAR_CONSEC_FRAMES:
                    tired_now = True
                eye_closed_counter = 0
            break   # process only first detected face

        # DeepFace every N s
        if now - last_inf >= EMOTION_SAMPLE_INTERVAL:
            try:
                out = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
                raw = out[0]["dominant_emotion"] if isinstance(out, list) else out["dominant_emotion"]
            except Exception:
                raw = "neutral"
            label = harmonise(raw, tired_now)
            emotion_window.append(label)
            print(f"[Facial] {int(now)}s -> {label}")
            last_inf = now

        cv2.imshow("Webcam (q quits)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cam.release(); cv2.destroyAllWindows()

# ───────── Keyboard / Mouse / Reports (unchanged) ─────────
def on_press(key): key_press_times[str(key)] = time.time()
def on_release(key):
    rel = time.time(); prs = key_press_times.pop(str(key), None)
    if prs: typing_metrics.append(rel - prs)
    if key == keyboard.Key.esc: return False
def on_scroll(x,y,dx,dy): global scroll_events; scroll_events += 1

def analyze_emotions():
    global emotion_window, typing_metrics, scroll_events
    while True:
        time.sleep(EMOTION_REPORT_INTERVAL)
        counts = Counter(e for e in emotion_window if e in NEGATIVE_EMOTIONS)
        for emo, cnt in counts.items():
            if cnt >= ALERT_THRESHOLD:
                threading.Thread(target=show_popup,args=(emo,),daemon=True).start()
                emotion_window = [e for e in emotion_window if e != emo]
        face_mode = max(set(emotion_window), key=emotion_window.count) if emotion_window else "engagement/focus"
        if typing_metrics:
            avg = statistics.mean(typing_metrics)
            typing_state = ("confused/frustrated" if avg > 0.5 else "confident" if avg < 0.15 else "neutral")
        else: typing_state = "inactive"
        scroll_state = "impatient/restless" if scroll_events > 10 else "focused"
        if face_mode == "engagement/focus" and scroll_events > 15: face_mode = "boredom"
        print("\n[30-second EMOTION REPORT]")
        print(f"Facial   : {face_mode}")
        print(f"Typing   : {typing_state}")
        print(f"Scrolling: {scroll_state}")
        print("-"*30,"\n")
        typing_metrics.clear(); scroll_events = 0

# ───────── Main launcher (unchanged) ───────
if __name__ == "__main__":
    cam_thread  = threading.Thread(target=detect_emotion,  daemon=True)
    fuse_thread = threading.Thread(target=analyze_emotions, daemon=True)
    kb_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    mouse_listener = mouse.Listener(on_scroll=on_scroll)

    cam_thread.start(); fuse_thread.start()
    kb_listener.start(); mouse_listener.start()

    kb_listener.join(); mouse_listener.stop()
    print("\nSession ended.")
