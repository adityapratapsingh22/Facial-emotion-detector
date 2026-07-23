from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import cv2
from fer import FER
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

detector = FER(mtcnn=True)

# Database setup
def get_db():
    db = sqlite3.connect("history.db")
    db.row_factory = sqlite3.Row
    return db

def init_db():
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            image_path TEXT,
            top_emotion TEXT,
            confidence TEXT,
            timestamp TEXT
        )
    """)
    db.commit()
    db.close()

init_db()

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/", methods=["GET"])
def index():
    db = get_db()
    history = db.execute(
        "SELECT * FROM detections ORDER BY id DESC LIMIT 20"
    ).fetchall()
    db.close()
    return render_template("index.html", result=None, image_path=None, history=history)

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("index"))

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    result = detector.detect_emotions(img)

    if len(result) == 0:
        return render_template("index.html", result="No face detected.", image_path=filepath, history=get_history())

    CONFIDENCE_THRESHOLD = 40.0
    faces_summary = []
    for face in result:
        emotions = face["emotions"]
        top_emotion = max(emotions, key=emotions.get)
        confidence = emotions[top_emotion] * 100

        if confidence < CONFIDENCE_THRESHOLD:
            faces_summary.append({
                "top_emotion": "unclear",
                "confidence": f"{confidence:.1f}",
                "all_emotions": emotions
            })
        else:
            faces_summary.append({
                "top_emotion": top_emotion,
                "confidence": f"{confidence:.1f}",
                "all_emotions": emotions
            })

    top = faces_summary[0]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    db = get_db()
    db.execute(
        "INSERT INTO detections (filename, image_path, top_emotion, confidence, timestamp) VALUES (?, ?, ?, ?, ?)",
        (file.filename, filepath, top["top_emotion"], top["confidence"], timestamp)
    )
    db.commit()
    db.close()

    return render_template("index.html", result=faces_summary, image_path=filepath, history=get_history())

def get_history():
    db = get_db()
    history = db.execute(
        "SELECT * FROM detections ORDER BY id DESC LIMIT 20"
    ).fetchall()
    db.close()
    return history

if __name__ == "__main__":
    app.run(debug=True)