import sounddevice as sd
import numpy as np
import datetime
import threading
import sqlite3
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
import subprocess

app = Flask(__name__)

# ======================================================
# CONFIGURATION
# ======================================================
SAMPLE_RATE = 44100
WINDOW_SECS = 30
CONFIDENCE_THRESHOLD = 0.5

LAT = 52.0   # Europe (placeholder - you can change)
LON = 5.0

DB_FILE = "detections.db"

# ======================================================
# DATABASE SETUP
# ======================================================
CSV_FILE = "data/output.csv"

# Helper: read all rows from CSV
def read_csv():
    rows = []
    try:
        with open(CSV_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("timestamp") or not row.get("species") or not row.get("confidence"):
                    continue  # skip invalid rows
                try:
                    row["timestamp"] = datetime.datetime.fromisoformat(row["timestamp"])
                    row["confidence"] = float(row["confidence"])
                    row["species"] = row["species"].strip()
                    rows.append(row)
                except Exception:
                    continue  # skip rows that fail parsing
    except FileNotFoundError:
        pass
    return rows

def save_detection(species, confidence):
    ts = datetime.datetime.now().isoformat()
    write_header = False
    try:
        with open(CSV_FILE, "r"):
            pass
    except FileNotFoundError:
        write_header = True
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp","species","confidence"])
        if write_header:
            writer.writeheader()
        writer.writerow({"timestamp": ts, "species": species, "confidence": confidence})

def save_detection_audio(indata, species, confidence):
    os.makedirs("detections", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_species = species.replace(" ", "_")

    filename = f"detections/{timestamp}_{safe_species}_{confidence:.2f}.wav"

    # Save the WAV with sample rate and single channel
    write(filename, SAMPLE_RATE, indata[:, 0])

    print(f"[AUDIO SAVED] {filename}")

def save_bird_audio(indata, start_sec, end_sec, species):
    # Convert seconds → sample indices
    start_i = int(start_sec * SAMPLE_RATE)
    end_i = int(end_sec * SAMPLE_RATE)

    segment = indata[start_i:end_i]  # slice the audio

    # Ensure output dir exists
    os.makedirs("detections", exist_ok=True)

    # File name
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"detections/{species}_{timestamp}.wav"

    # Save WAV
    write(filename, SAMPLE_RATE, segment.astype(np.float32))
    print(f"Saved bird segment → {filename}")

# ======================================================
# BIRDNEN ANALYZER INIT
# ======================================================
MODEL_PATH = "/System/Volumes/Data/Users/niko/Library/Application Support/birdnet/acoustic-models/v2.4/tf/model-fp32.tflite"
LABELS_PATH = "/System/Volumes/Data/Users/niko/Library/Application Support/birdnet/acoustic-models/v2.4/tf/labels/nl.txt"

analyzer = Analyzer(
    classifier_model_path=MODEL_PATH,
    classifier_labels_path=LABELS_PATH
)

# ======================================================
# REAL-TIME AUDIO CALLBACK
# ======================================================
def audio_callback(indata, frames, time_info, status):
    audio = np.squeeze(indata)
    now = datetime.datetime.now().strftime("%Y-%m-%d")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        from scipy.io.wavfile import write
        write(temp_audio.name, SAMPLE_RATE, indata[:, 0])
        # Write the same audio data to another disk
        with open("recordings/audio_" + now + ".wav", "wb") as f:
            write(f, SAMPLE_RATE, indata[:, 0])
        rec = Recording(path=temp_audio.name, analyzer=analyzer)

    rec.analyze()

    if rec.detections:
        for r in rec.detections:
            species = r["common_name"]
            conf = r["confidence"]

            # Skip human vocalizations
            if species.lower() == "human vocal":
                continue
            if conf >= CONFIDENCE_THRESHOLD:
                print(f"{species}: {conf:.2f}")

                # Extract only the bird segment
                start_time = r["start_time"]
                end_time = r["end_time"]

                save_bird_audio(audio, start_time, end_time, species)

                save_detection(species, conf)

                # NEW: Save audio snippet
                # save_detection_audio(indata, species, conf)
    else:
        print("No detections above threshold.")

# ======================================================
# THREAD FOR AUDIO LISTENING
# ======================================================
def start_listener():
    with sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=SAMPLE_RATE * WINDOW_SECS,
        dtype="float32",
        callback=audio_callback
    ):
        print("🔊 BirdNET is listening... dashboard: http://localhost:4000")
        while True:
            time.sleep(1)

listener_thread = threading.Thread(target=start_listener)
listener_thread.daemon = True
listener_thread.start()

# ======================================================
# Gitcommiter thread to auto-commit CSV every hour
# ======================================================


def auto_git_committer(fileName = CSV_FILE):
    last_commit_hour = None

    while True:
        now = datetime.datetime.now()

        # Run at the exact start of the hour
        if now.minute == 1 and now.second == 0:
            hour_key = now.strftime("%Y%m%d%H")

            if hour_key != last_commit_hour:
                print(f"[GIT] Auto-committing output.csv at {now}")

                try:
                    # Stage file
                    subprocess.run(["git", "add", fileName], check=True)

                    # Commit
                    msg = f"Auto-commit output.csv at {now.strftime('%Y-%m-%d %H:%M')}"
                    subprocess.run(["git", "commit", "-m", msg], check=True)

                    # OPTIONAL: push to remote
                    # subprocess.run(["git", "push"], check=True)

                    last_commit_hour = hour_key

                except subprocess.CalledProcessError as e:
                    print(f"[GIT ERROR] {e}")

        time.sleep(1)  # check once per second

# Start git commit thread
git_thread = threading.Thread(target=auto_git_committer)
git_thread.daemon = True
git_thread.start()

# ======================================================
# DASHBOARD WEB SERVER
# ======================================================

# ======================================================
# HTML TEMPLATE
# ======================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>BirdNET Dashboard</title>
    <meta http-equiv="refresh" content="60">
    <style>
        body { font-family: Arial; margin: 40px; }
        h1 { color: #2c6e49; }
        .box { border: 1px solid #ddd; padding: 20px; margin: 20px 0; }
        img { max-width: 900px; width: 100%; }
    </style>
</head>
<body>

<h1>🐦 BirdNET Real-Time Dashboard</h1>
<p>Auto-refreshes every 10 seconds.</p>

<h2>📈 Summary</h2>
<div class="box">
    <p><strong>Last 24 hours:</strong> {{ s24 }}</p>
    <p><strong>Last 48 hours:</strong> {{ s48 }}</p>
    <p><strong>Last 7 days:</strong> {{ s7 }}</p>
    <p><strong>Last 30 days:</strong> {{ s30 }}</p>
</div>

<h2>🔥 Latest Detections</h2>
<div class="box">
{% for d in latest %}
    <p>{{ d["timestamp"].strftime("%Y-%m-%d %H:%M:%S") }} — {{ d["species"] }} ({{ "%.2f"|format(d["confidence"]) }})</p>
{% endfor %}
</div>

<h2>🏆 Top Species (7 days)</h2>
<div class="box">
{% for s, c in top_species %}
    <p><strong>{{ s }}</strong> — {{ c }} detections</p>
{% endfor %}
</div>

<h2>📊 Average Hourly Activity (all birds)</h2>
<div class="box">
    <img src="/activity.png?{{ now }}">
</div>

# <h2>📊 Hourly Activity for Top 10 Birds</h2>
# <div class="box">
# {% for s, c in top_species %}
#     <h3>{{ s }}</h3>
#     <img src="/species_activity/{{ s }}.png?{{ now }}">
# {% endfor %}
# </div>

</body>
</html>
"""

# ======================================================
# COUNTING HELPERS
# ======================================================

def count_since(hours):
    cutoff = datetime.datetime.now() - datetime.timedelta(hours=hours)
    rows = read_csv()
    return sum(1 for r in rows if r["timestamp"] > cutoff)

def get_latest():
    rows = read_csv()
    rows_sorted = sorted(rows, key=lambda x: x["timestamp"], reverse=True)
    return rows_sorted[:20]

def get_top_species(days=7):
    cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
    rows = read_csv()
    filtered = [r for r in rows if r["timestamp"] > cutoff]
    counter = Counter(r["species"] for r in filtered if r["species"])
    return counter.most_common(10)


# ======================================================
# HOURLY ACTIVITY HEATMAP CALCULATOR
# ======================================================

def hourly_activity(rows, days=7):
    cutoff = datetime.datetime.now() - datetime.timedelta(days=days)

    heatmap = [[0 for _ in range(24)] for _ in range(days)]

    for r in rows:
        ts = r["timestamp"]
        if ts > cutoff:
            age = (datetime.datetime.now() - ts).days
            if age < days:
                heatmap[days - age - 1][ts.hour] += 1

    return heatmap

# ======================================================
# HOURLY CACHE SYSTEM
# ======================================================

plot_cache = {
    "activity": {"timestamp": None, "image": None},
    "species": {}
}

def hour_key():
    """Return YYYYMMDDHH as cache key."""
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d%H")

# ======================================================
# ROUTE: AVERAGE ACTIVITY HEATMAP
# ======================================================

@app.route("/activity.png")
def activity_png():
    key = hour_key()

    # Return cached image if still valid
    if plot_cache["activity"]["timestamp"] == key:
        return send_file(
            io.BytesIO(plot_cache["activity"]["image"]),
            mimetype="image/png"
        )

    # Otherwise regenerate
    rows = read_csv()
    data = hourly_activity(rows)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(data, aspect="auto")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Days Ago")
    ax.set_xticks(range(24))
    ax.set_yticks(range(len(data)))

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_bytes = buf.getvalue()
    plt.close(fig)

    # Store in cache
    plot_cache["activity"] = {
        "timestamp": key,
        "image": img_bytes
    }

    return send_file(io.BytesIO(img_bytes), mimetype="image/png")

# ======================================================
# ROUTE: SPECIES HEATMAP
# ======================================================

# @app.route("/species_activity/<species>.png")
# def species_activity_png(species):

#     key = hour_key()

#     # Initialize cache bucket for species
#     if species not in plot_cache["species"]:
#         plot_cache["species"][species] = {"timestamp": None, "image": None}

#     # Serve cached
#     if plot_cache["species"][species]["timestamp"] == key:
#         return send_file(
#             io.BytesIO(plot_cache["species"][species]["image"]),
#             mimetype="image/png"
#         )

#     # Generate new image
#     rows = read_csv()
#     filtered = [r for r in rows if r["species"] == species]
#     data = hourly_activity(filtered)
#     fig, ax = plt.subplots(figsize=(10, 3))
#     ax.imshow(data, aspect="auto")
#     ax.set_xlabel("Hour of Day")
#     ax.set_ylabel("Days Ago")
#     ax.set_xticks(range(24))
#     ax.set_yticks(range(len(data)))

#     buf = io.BytesIO()
#     plt.savefig(buf, format="png", bbox_inches="tight")
#     plt.close(fig)
#     buf.seek(0)

#     img_bytes = buf.getvalue()

#     # Store cache
#     plot_cache["species"][species] = {
#         "timestamp": key,
#         "image": img_bytes
#     }

#     return send_file(io.BytesIO(img_bytes), mimetype="image/png")

# ======================================================
# MAIN DASHBOARD
# ======================================================

@app.route("/")
def dashboard():
    return render_template_string(
        HTML_TEMPLATE,
        s24=count_since(24),
        s48=count_since(48),
        s7=count_since(24*7),
        s30=count_since(24*30),
        latest=get_latest(),
        top_species=get_top_species(),
        now=datetime.datetime.now()
    )

# ======================================================
# START SERVER
# ======================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000)