import cv2
import csv
import os
import time
import mediapipe as mp

MODEL_PATH = "hand_landmarker.task"
DATASET_DIR = "dataset"

GESTURES = {
    "1": "hello",
    "2": "yes",
    "3": "no",
    "4": "sorry",
    "5": "please"
}

SEQUENCE_LENGTH = 30
SAMPLES_PER_GESTURE = 100

os.makedirs(DATASET_DIR, exist_ok=True)

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions


def create_landmarker():

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    return HandLandmarker.create_from_options(options)


def extract_landmarks(result):

    features = []

    # Always reserve space for 2 hands.
    for hand_index in range(2):

        if hand_index < len(result.hand_landmarks):

            landmarks = result.hand_landmarks[hand_index]

            wrist = landmarks[0]

            for landmark in landmarks:

                features.append(landmark.x - wrist.x)
                features.append(landmark.y - wrist.y)
                features.append(landmark.z - wrist.z)

        else:

            # No second hand detected.
            features.extend([0.0] * 63)

    return features


def collect_gesture(landmarker, gesture_name):

    filename = os.path.join(
        DATASET_DIR,
        f"{gesture_name}.csv"
    )

    print()
    print("=" * 60)
    print(f"Collecting ASL data for: {gesture_name}")
    print(f"Sequences required: {SAMPLES_PER_GESTURE}")
    print(f"Frames per sequence: {SEQUENCE_LENGTH}")
    print()
    print("Press SPACE to start.")
    print("Press Q to cancel.")
    print("=" * 60)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("ERROR: Could not open webcam.")
        return

    sequence = []
    sequence_count = 0
    started = False

    with open(filename, "w", newline="") as file:

        writer = csv.writer(file)

        while cap.isOpened():

            success, frame = cap.read()

            if not success:

                print("Could not read webcam frame.")
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

            timestamp = int(time.time() * 1000)

            result = landmarker.detect_for_video(
                mp_image,
                timestamp
            )

            landmarks = extract_landmarks(result)

            if started:

                sequence.append(landmarks)

                if len(sequence) == SEQUENCE_LENGTH:

                    # Store one complete movement sequence.
                    row = []

                    for frame_landmarks in sequence:

                        row.extend(frame_landmarks)

                    row.append(gesture_name)

                    writer.writerow(row)

                    sequence_count += 1
                    sequence = []

            cv2.putText(
                frame,
                f"Gesture: {gesture_name}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Sequences: {sequence_count}/{SAMPLES_PER_GESTURE}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            if not started:

                cv2.putText(
                    frame,
                    "Press SPACE to start",
                    (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "Perform the sign naturally",
                    (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

            cv2.imshow(
                "SignBridge - ASL Data Collection",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):

                started = True
                sequence = []

                print(
                    f"Started collecting {gesture_name}..."
                )

            elif key == ord("q"):

                break

            if sequence_count >= SAMPLES_PER_GESTURE:

                break

    cap.release()
    cv2.destroyAllWindows()

    print()
    print(
        f"Finished {gesture_name}: "
        f"{sequence_count} sequences collected."
    )


def main():

    if not os.path.exists(MODEL_PATH):

        print(
            "ERROR: hand_landmarker.task "
            "was not found."
        )

        return

    print()
    print("=" * 45)
    print("       SIGNBRIDGE ASL DATA COLLECTOR")
    print("=" * 45)

    print()
    print("Available ASL signs:")

    for key, gesture in GESTURES.items():

        print(f"{key}. {gesture}")

    choice = input(
        "\nSelect gesture number: "
    ).strip()

    if choice not in GESTURES:

        print("Invalid choice.")
        return

    gesture_name = GESTURES[choice]

    with create_landmarker() as landmarker:

        collect_gesture(
            landmarker,
            gesture_name
        )


if __name__ == "__main__":

    main()