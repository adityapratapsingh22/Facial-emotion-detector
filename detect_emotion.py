import cv2
from fer import FER

image_path = "face.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"ERROR: Could not load image at '{image_path}'. Check the filename and that it's in this folder.")
else:
    detector = FER(mtcnn=False)
    result = detector.detect_emotions(img)

    if len(result) == 0:
        print("No face detected in the image.")
    else:
        for face in result:
            print("Face box:", face["box"])
            print("Emotions:", face["emotions"])
            top_emotion = max(face["emotions"], key=face["emotions"].get)
            print(f"Dominant emotion: {top_emotion} ({face['emotions'][top_emotion]*100:.1f}%)")