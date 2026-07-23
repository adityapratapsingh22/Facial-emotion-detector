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

detector = FER(mtcnn=True)
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

    history.insert(0, {
        "filename": file.filename,
        "image_path": filepath,
        "top_emotion": faces_summary[0]["top_emotion"],
        "confidence": faces_summary[0]["confidence"]
    })

    return render_template("index.html", result=faces_summary, image_path=filepath, history=history)

if __name__ == "__main__":
    app.run(debug=True)