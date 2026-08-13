from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time


app = Flask(__name__)

# --------------------------------------------------
# Paths
# --------------------------------------------------

MODEL_PATH = "model/sign_model.keras"
LABELS_PATH = "model/labels.txt"
HAND_MODEL_PATH = "hand_landmarker.task"

# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as file:
    labels = [line.strip() for line in file if line.strip()]

print("Loaded labels:", labels)


# --------------------------------------------------
# MediaPipe Hand Landmarker
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions


def create_landmarker():

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=HAND_MODEL_PATH
        ),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    return HandLandmarker.create_from_options(options)


# --------------------------------------------------
# Extract landmarks
# --------------------------------------------------

def extract_landmarks(result):

    features = []

    # Reserve space for 2 hands.
    for hand_index in range(2):

        if hand_index < len(result.hand_landmarks):

            landmarks = result.hand_landmarks[hand_index]

            wrist = landmarks[0]

            for landmark in landmarks:

                features.append(
                    landmark.x - wrist.x
                )

                features.append(
                    landmark.y - wrist.y
                )

                features.append(
                    landmark.z - wrist.z
                )

        else:

            # No hand detected.
            features.extend([0.0] * 63)

    return features


# --------------------------------------------------
# Webcam generator
# --------------------------------------------------

def generate_frames():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("ERROR: Could not open webcam.")
        return

    sequence = []

    last_prediction = "WAITING"
    confidence = 0.0

    with create_landmarker() as landmarker:

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = int(
                time.time() * 1000
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp
            )

            # ------------------------------------------
            # Hand detected
            # ------------------------------------------

            if result.hand_landmarks:

                landmarks = extract_landmarks(
                    result
                )

                sequence.append(landmarks)

                if len(sequence) > 30:

                    sequence = sequence[-30:]

                # Draw landmarks
                for hand_landmarks in result.hand_landmarks:

                    for landmark in hand_landmarks:

                        x = int(
                            landmark.x * frame.shape[1]
                        )

                        y = int(
                            landmark.y * frame.shape[0]
                        )

                        cv2.circle(
                            frame,
                            (x, y),
                            4,
                            (0, 255, 0),
                            -1
                        )

                # --------------------------------------
                # Predict after 30 frames
                # --------------------------------------

                if len(sequence) == 30:

                    input_data = np.array(
                        sequence,
                        dtype=np.float32
                    )

                    input_data = np.expand_dims(
                        input_data,
                        axis=0
                    )

                    prediction = model.predict(
                        input_data,
                        verbose=0
                    )[0]

                    predicted_index = int(
                        np.argmax(prediction)
                    )

                    confidence = float(
                        prediction[predicted_index]
                    )

                    if predicted_index < len(labels):

                        last_prediction = labels[
                            predicted_index
                        ].upper()

            else:

                sequence = []

                last_prediction = "NO HAND"
                confidence = 0.0

            # ------------------------------------------
            # Display
            # ------------------------------------------

            cv2.rectangle(
                frame,
                (15, 20),
                (450, 115),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                f"Sign: {last_prediction}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence * 100:.1f}%",
                (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            ret, buffer = cv2.imencode(
                ".jpg",
                frame
            )

            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

    cap.release()


# --------------------------------------------------
# Flask routes
# --------------------------------------------------

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )
@app.route("/translate", methods=["GET", "POST"])
def translate():
    return render_template("sign.html")

# --------------------------------------------------
# Run application
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=False,
        host="127.0.0.1",
        port=5000
    )