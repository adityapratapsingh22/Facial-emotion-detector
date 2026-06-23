from flask import Flask, request, render_template, redirect, send_from_directory, url_for
import cv2
from fer import FER
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

detector = FER(mtcnn=False)
history = []

@app.route("/", methods=["GET"])
def index():
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
        return render_template("index.html", result="No face detected.", image_path=filepath, history=history)

    face = result[0]
    emotions = face["emotions"]
    top_emotion = max(emotions, key=emotions.get)
    confidence = emotions[top_emotion] * 100

    summary = {
        "top_emotion": top_emotion,
        "confidence": f"{confidence:.1f}",
        "all_emotions": emotions
    }

    history.insert(0, {
        "filename": file.filename,
        "image_path": filepath,
        "top_emotion": top_emotion,
        "confidence": f"{confidence:.1f}"
    })

    return render_template("index.html", result=summary, image_path=filepath, history=history)

if __name__ == "__main__":
    app.run(debug=True)