import sounddevice as sd
import numpy as np
import datetime
import threading
import time
from flask import Flask, render_template_string, send_file
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import io
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
import tempfile
import os
import csv
from collections import Counter
from scipy.io.wavfile import write
import shutil
import subprocess

app = Flask(__name__)

# ======================================================
# CONFIG
# ======================================================
SAMPLE_RATE = 44100
WINDOW_SECS = 30
CONFIDENCE_THRESHOLD = 0.5

CSV_FILE = "../data/output.csv"
DETECTIONS_DIR = "detections"

MAX_AUDIO_FILES = 200
MIN_FREE_SPACE_GB = 5
MAX_CSV_SIZE_MB = 50

# ======================================================
# DISK SAFETY
# ======================================================

def disk_ok():
    total, used, free = shutil.disk_usage("/")
    return free > MIN_FREE_SPACE_GB * 1024**3

def cleanup_old_audio():
    if not os.path.exists(DETECTIONS_DIR):
        return

    files = sorted(
        [os.path.join(DETECTIONS_DIR, f) for f in os.listdir(DETECTIONS_DIR)],
        key=os.path.getmtime
    )

    for f in files[:-MAX_AUDIO_FILES]:
        try:
            os.remove(f)
        except:
            pass

# ======================================================
# CSV STORAGE + ROTATION
# ======================================================

def rotate_csv():
    if not os.path.exists(CSV_FILE):
        return

    size_mb = os.path.getsize(CSV_FILE) / (1024 * 1024)

    if size_mb > MAX_CSV_SIZE_MB:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        archive = f"../data/archive_{timestamp}.csv"
        os.rename(CSV_FILE, archive)
        print(f"[CSV] Rotated → {archive}")

def read_csv():
    rows = []
    try:
        with open(CSV_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    row["timestamp"] = datetime.datetime.fromisoformat(row["timestamp"])
                    row["confidence"] = float(row["confidence"])
                    rows.append(row)
                except:
                    continue
    except FileNotFoundError:
        pass
    return rows

def save_detection(species, confidence):
    ts = datetime.datetime.now().isoformat()
    write_header = not os.path.exists(CSV_FILE)

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp","species","confidence"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": ts,
            "species": species,
            "confidence": confidence
        })

# ======================================================
# OPTIONAL AUDIO SAVE (SAFE)
# ======================================================

def save_detection_audio(indata, species, confidence):
    if not disk_ok():
        print("[WARNING] Low disk space — skipping audio")
        return

    os.makedirs(DETECTIONS_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_species = species.replace(" ", "_")

    filename = f"{DETECTIONS_DIR}/{timestamp}_{safe_species}_{confidence:.2f}.wav"

    write(filename, SAMPLE_RATE, indata[:, 0])
    cleanup_old_audio()

    print(f"[AUDIO SAVED] {filename}")

# ======================================================
# BIRDNET
# ======================================================

MODEL_PATH = "/System/Volumes/Data/Users/niko/Library/Application Support/birdnet/acoustic-models/v2.4/tf/model-fp32.tflite"
LABELS_PATH = "/System/Volumes/Data/Users/niko/Library/Application Support/birdnet/acoustic-models/v2.4/tf/labels/nl.txt"

analyzer = Analyzer(
    classifier_model_path=MODEL_PATH,
    classifier_labels_path=LABELS_PATH
)

# ======================================================
# AUDIO CALLBACK
# ======================================================

def audio_callback(indata, frames, time_info, status):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio:
        write(temp_audio.name, SAMPLE_RATE, indata[:, 0])

        rec = Recording(path=temp_audio.name, analyzer=analyzer)
        rec.analyze()

    if rec.detections:
        for r in rec.detections:
            species = r["common_name"]
            conf = r["confidence"]

            if species.lower() == "human vocal":
                continue

            if conf >= CONFIDENCE_THRESHOLD:
                print(f"{species}: {conf:.2f}")

                save_detection(species, conf)

                # OPTIONAL:
                # save_detection_audio(indata, species, conf)
    else:
        print("No detections")

# ======================================================
# AUDIO THREAD
# ======================================================

def start_listener():
    with sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=SAMPLE_RATE * WINDOW_SECS,
        dtype="float32",
        callback=audio_callback
    ):
        print("🔊 Listening... http://localhost:4000")
        while True:
            time.sleep(1)

threading.Thread(target=start_listener, daemon=True).start()

# ======================================================
# GIT AUTO COMMIT (SAFE)
# ======================================================

def auto_git_committer():
    last_commit_hour = None

    while True:
        now = datetime.datetime.now()
        current_hour = now.strftime("%Y%m%d%H")

        if current_hour != last_commit_hour:
            try:
                rotate_csv()

                result = subprocess.run(
                    ["git", "status", "--porcelain", CSV_FILE],
                    capture_output=True,
                    text=True
                )

                if result.stdout.strip() == "":
                    print("[GIT] No changes")
                else:
                    print("[GIT] Committing CSV")

                    subprocess.run(["git", "add", CSV_FILE], check=True)

                    msg = f"Auto-commit {now.strftime('%Y-%m-%d %H:%M')}"
                    subprocess.run(["git", "commit", "-m", msg], check=True)

                    subprocess.run(["git", "push"], check=True)

                    subprocess.run(["git", "gc", "--auto"], check=True)

                last_commit_hour = current_hour

            except subprocess.CalledProcessError as e:
                print(f"[GIT ERROR] {e}")

        time.sleep(60)

threading.Thread(target=auto_git_committer, daemon=True).start()

# ======================================================
# DASHBOARD
# ======================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="60">
</head>
<body>
<h1>BirdNET Dashboard</h1>
<p>Last 24h: {{ s24 }}</p>
</body>
</html>
"""

def count_since(hours):
    cutoff = datetime.datetime.now() - datetime.timedelta(hours=hours)
    return sum(1 for r in read_csv() if r["timestamp"] > cutoff)

@app.route("/")
def dashboard():
    return render_template_string(
        HTML_TEMPLATE,
        s24=count_since(24)
    )

# ======================================================
# RUN
# ======================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000)
