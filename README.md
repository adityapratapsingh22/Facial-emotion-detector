# Facial Emotion Detector

A web application that detects emotions from facial expressions using a pretrained deep learning model. Supports both image upload and live webcam capture, with a results history panel.

## Features

- **Image upload** — upload any photo with a visible face and get an emotion breakdown
- **Live webcam capture** — capture a photo directly from your browser's camera and detect emotion in real time
- **Detailed results** — shows the dominant emotion plus a full breakdown across all 7 categories (angry, disgust, fear, happy, sad, surprise, neutral)
- **Detection history** — tracks recent uploads and webcam captures in-session

## Tech Stack

- **Backend:** Python, Flask
- **Computer vision / ML:** OpenCV (face detection), [FER](https://github.com/justinshenk/fer) (pretrained CNN for emotion classification, built on TensorFlow/Keras)
- **Frontend:** HTML, CSS, vanilla JavaScript (webcam access via `getUserMedia`)

## How It Works

1. OpenCV locates a face in the input image (Haar Cascade detector)
2. The cropped face is passed to a pretrained CNN (trained on the FER2013 dataset), which outputs a probability score for each of 7 emotions
3. Flask renders the result back to the browser, including a visual breakdown of all emotion scores
4. For webcam input, a frame is captured client-side, converted to an image blob, and sent to the same backend endpoint as a regular file upload

## Setup

```bash
# Clone the repo
git clone https://github.com/adityapratapsingh22/Facial-emotion-detector.git
cd Facial-emotion-detector

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install opencv-python fer==22.5.1 flask flask-cors tensorflow moviepy

# Run the app
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

> **Note:** This project pins `fer==22.5.1` specifically — newer releases of the `fer` package on PyPI have a broken import (`ImportError: cannot import name 'FER'`), so this version is required for the app to run.

## Limitations

- Facial emotion recognition from a single static image is inherently imprecise. The underlying model is trained on the FER2013 dataset, which has known labeling inconsistencies — even published benchmarks on this dataset typically report ~65-75% accuracy.
- Detection quality depends heavily on lighting, face angle, and image clarity. Real-world accuracy on arbitrary photos will vary.
- This project is built for demonstration and learning purposes, not as a clinically or commercially reliable emotion-detection tool.
- Detection history resets when the Flask server restarts (stored in memory, not a database).
- Runs in Flask's development server — not configured for production deployment.

## Project Structure

```
.
├── app.py                  # Flask application and routes
├── detect_emotion.py       # Standalone script for testing detection on a single image
├── test_setup.py           # Environment verification script
├── templates/
│   └── index.html          # Frontend (upload form, webcam capture, results display)
├── uploads/                 # Stores uploaded/captured images (gitignored)
└── requirements (see Setup above)
```

## Author

Aditya Pratap Singh
