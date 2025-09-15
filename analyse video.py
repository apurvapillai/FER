"""
Pure-inference emotion analysis (no training, no calibration), NO MediaPipe.
- Face detection: OpenCV Haar cascade (bundled with opencv-python)
- 6 emotions via FER CNN: angry, disgust, fear, happy, sad, surprise
- Contempt via a deterministic mouth-asymmetry heuristic (no landmarks)
- Neutral if none above confident

Outputs a CSV with segments (label, start_time, end_time, max_conf) and prints
an ordered list of detected emotions.

Install:
  pip install opencv-python fer numpy pandas yt-dlp

Usage:
  python analyze_video.py --video "https://youtu.be/..." --show
  python analyze_video.py --video path/to/local.mp4 --log out.csv
"""

import os, sys, argparse, tempfile, subprocess
import cv2, numpy as np, pandas as pd
from collections import defaultdict

from fer import FER

CONF_THRESH = 0.40        # accept FER emotion >= this
NEUTRAL_THRESH = 0.30     # if all FER probs < this => neutral candidate
CONTEMPT_ASYM_T = 0.08    # smirk threshold from mouth asymmetry (0.06–0.12)
MIN_FACE = 90             # px
PAD_FRAC = 0.20           # pad around detected face
DOWNSCALE = 1.0           # set 0.75 or 0.5 if you need speed
MIN_SEG_FRAMES = 2        # min frames per segment unless spike
SPIKE_CONF = 0.60         # count 1-frame spikes if high enough
DETECT_EVERY = 1          # Haar is cheap; detect every frame for snappier changes

DISPLAY_ORDER = ["Surprise","Happiness","Anger","Disgust","Contempt","Sadness","Fear","Neutral"]
COLOR = {
    "Anger": (0,0,255), "Disgust": (0,128,128), "Fear": (128,0,128),
    "Happiness": (0,255,0), "Sadness": (255,0,0), "Surprise": (0,255,255),
    "Contempt": (200,200,200), "Neutral": (180,180,255)
}
FER2DISP = {
    "angry":"Anger","disgust":"Disgust","fear":"Fear",
    "happy":"Happiness","sad":"Sadness","surprise":"Surprise"
}
FER_EMOS = set(FER2DISP.keys())

classifier = FER(mtcnn=False)
HAAR_FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def fetch_video_path(video_arg: str) -> str:
    if os.path.exists(video_arg):
        return video_arg
    # assume URL -> download to temp
    out_dir = tempfile.mkdtemp(prefix="ytvid_")
    out_path = os.path.join(out_dir, "video.mp4")
    cmd = ["yt-dlp", "-f", "mp4", "-o", out_path, video_arg]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return out_path
    except subprocess.CalledProcessError as e:
        print("yt-dlp failed:", e.stderr.decode("utf-8", errors="ignore"))
        sys.exit(1)

def pad_box(x, y, w, h, W, H, frac=PAD_FRAC):
    px, py = int(frac*w), int(frac*h)
    x1 = max(0, x - px); y1 = max(0, y - py)
    x2 = min(W - 1, x + w + px); y2 = min(H - 1, y + h + py)
    return x1, y1, x2, y2

def fer_top(rgb_crop):
    label, score = classifier.top_emotion(rgb_crop)
    if label is None or score is None:
        return "unknown", 0.0, {}
    dets = classifier.detect_emotions(rgb_crop)
    probs = dets[0]["emotions"] if dets else {}
    return label, float(score), probs

def contempt_score_mouth_asym(bgr_crop):
  
    if bgr_crop is None or bgr_crop.size == 0:
        return 0.0
    h, w = bgr_crop.shape[:2]
    if w < 40 or h < 40:
        return 0.0

    # Lower third as mouth ROI
    y0 = int(h * 2/3)
    roi = bgr_crop[y0:h, :]

    # Preprocess
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    # Edge map
    edges = cv2.Canny(gray, 50, 150)

    # For each column, find top-most edge (approx upper lip edge)
    ys = []
    for x in range(edges.shape[1]):
        col = edges[:, x].nonzero()[0]
        ys.append(col[0] if len(col) else edges.shape[0])  # if no edge, bottom

    ys = np.asarray(ys, dtype=np.float32)
    if ys.size < 10:
        return 0.0

    # Split left/right halves
    mid = ys.size // 2
    left_mean = float(np.mean(ys[:mid]))
    right_mean = float(np.mean(ys[mid:]))

    # Asymmetry: difference in height (smaller y = higher lip corner)
    asym = abs(left_mean - right_mean) / (edges.shape[0] + 1e-6)

    # Mouth openness proxy: average distance from top edge to bottom
    # If many columns have no edge (bottom), openness will look large -> damp contempt
    openness = float(np.mean(ys)) / (edges.shape[0] + 1e-6)
    damp = 1.0 - np.clip(openness, 0.0, 1.0)

    score = np.clip(asym * damp, 0.0, 1.0)
    return float(score)

class Segmenter:
    """Non-adaptive segmentation; preserves short spikes above SPIKE_CONF."""
    def __init__(self, fps):
        self.fps = fps
        self.cur_label = None
        self.cur_conf = 0.0
        self.start = 0
        self.frame = 0
        self.seg = []

    def update(self, label, conf):
        f = self.frame
        if self.cur_label is None:
            self.cur_label, self.cur_conf, self.start = label, conf, f
        elif label == self.cur_label:
            self.cur_conf = max(self.cur_conf, conf)
        else:
            length = f - self.start
            if length >= MIN_SEG_FRAMES or self.cur_conf >= SPIKE_CONF:
                self.seg.append({
                    "label": self.cur_label,
                    "start_frame": self.start,
                    "end_frame": f-1,
                    "max_conf": self.cur_conf
                })
            self.cur_label, self.cur_conf, self.start = label, conf, f
        self.frame += 1

    def finalize(self):
        f = self.frame
        if self.cur_label is not None:
            length = f - self.start
            if length >= MIN_SEG_FRAMES or self.cur_conf >= SPIKE_CONF:
                self.seg.append({
                    "label": self.cur_label,
                    "start_frame": self.start,
                    "end_frame": f-1,
                    "max_conf": self.cur_conf
                })
        for s in self.seg:
            s["start_time"] = s["start_frame"] / self.fps
            s["end_time"]   = s["end_frame"]   / self.fps
        return self.seg

def analyze(video_arg: str, show: bool, out_csv: str):
    path = fetch_video_path(video_arg)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Could not open video."); sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    seg = Segmenter(fps)

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        if DOWNSCALE != 1.0: #Optionally downscale (for speed).Record height H and width W.
            frame = cv2.resize(frame, None, fx=DOWNSCALE, fy=DOWNSCALE, interpolation=cv2.INTER_AREA)
        H, W = frame.shape[:2]

        # Detect faces (largest face)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert to grayscale (Haar works better on gray). Detect faces. If no faces → mark as Neutral:
        faces = HAAR_FACE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE, MIN_FACE)
        )
        if len(faces) == 0:
            seg.update("Neutral", 0.0)
            if show:
                cv2.imshow("Emotion Analysis", frame)
                if cv2.waitKey(1) & 0xFF == 27: break
            frame_idx += 1
            continue

        x, y, w, h = max(faces, key=lambda b: b[2]*b[3])  # largest
        x1, y1, x2, y2 = pad_box(x, y, w, h, W, H, PAD_FRAC)
        crop = frame[y1:y2, x1:x2]
        rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # FER inference
        fer_label, fer_conf, fer_probs = fer_top(rgb)

        # Contempt vs Neutral decision (heuristic)
        contempt_conf = contempt_score_mouth_asym(crop)
        neutral_candidate = (fer_label == "unknown") or all(fer_probs.get(k,0.0) < NEUTRAL_THRESH for k in FER_EMOS)

        # Final label
        if fer_label in FER_EMOS and fer_conf >= CONF_THRESH:
            label = FER2DISP[fer_label]; conf = fer_conf
        else:
            if contempt_conf >= CONTEMPT_ASYM_T:
                label, conf = "Contempt", contempt_conf
            else:
                label, conf = "Neutral", max(0.0, 1.0 - max(fer_probs.values()) if fer_probs else 0.5)

        # Update segments
        seg.update(label, conf)

        # Draw (optional)
        if show:
            cv2.rectangle(frame, (x1,y1), (x2,y2), COLOR.get(label,(0,255,0)), 2)
            cv2.putText(frame, f"{label} {conf*100:.1f}%", (x1, max(0,y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR.get(label,(0,255,0)), 2)
            t = frame_idx / fps
            cv2.putText(frame, f"{t:6.2f}s", (10, H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 2)
            cv2.imshow("Emotion Analysis", frame)
            if cv2.waitKey(1) & 0xFF == 27: break

        frame_idx += 1

    cap.release(); cv2.destroyAllWindows()

    # Save & print results
    segments = seg.finalize()
    if not segments:
        print("No segments detected."); return
    df = pd.DataFrame(segments, columns=["label","start_frame","end_frame","start_time","end_time","max_conf"])
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    print("\nOrdered list of emotions (start time):")
    for s in segments:
        print(f" - {s['start_time']:.2f}s: {s['label']}")

    print("\nCounts:")
    print(df["label"].value_counts().to_string())

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="YouTube URL or local video path")
    p.add_argument("--log", default="emotions_log.csv", help="Output CSV")
    p.add_argument("--show", action="store_true", help="Show annotated video window")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    analyze(args.video, show=args.show, out_csv=args.log)
