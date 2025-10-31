from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(_name_)
model = load_model('model/sign_model.h5')
labels = ['Hello', 'Thanks', 'Yes', 'No', 'I Love You']

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def generate_frames():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        prediction = np.random.choice(labels)
                        cv2.putText(image, f'Sign: {prediction}', (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/translate', methods=['GET', 'POST'])
def translate():
    if request.method == 'POST':
        text = request.form.get('text')
        translated = text[::-1]  # Mock translation logic
        return render_template('translate.html', translated=translated)
    return render_template('translate.html', translated=None)

if _name_ == '_main_':
    app.run(debug=True)
