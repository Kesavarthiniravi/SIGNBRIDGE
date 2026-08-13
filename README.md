# SignBridge AI

## AI-Powered Real-Time Sign Language Recognition

SignBridge AI is a computer vision and deep learning application designed to recognize selected American Sign Language (ASL) gestures in real time through a webcam.

The system uses MediaPipe to extract hand landmarks from video frames and an LSTM-based deep learning model to analyze sequences of hand movements and generate real-time predictions.

The project is designed as an extensible prototype that can be expanded with additional gesture classes and larger datasets.

---

## Features

- Real-time webcam-based sign recognition
- Hand landmark detection using MediaPipe
- Sequence-based gesture recognition
- LSTM-based deep learning model
- Real-time prediction with confidence score
- Flask-based web application
- Interactive webcam interface
- Modular dataset and model pipeline
- Extensible architecture for additional gestures

---

## Technology Stack

- Python
- Flask
- OpenCV
- MediaPipe
- TensorFlow / Keras
- Scikit-learn
- NumPy
- HTML
- CSS

---

## System Architecture

```text
Webcam
   │
   ▼
Video Frame Capture
   │
   ▼
MediaPipe Hand Landmark Detection
   │
   ▼
Landmark Feature Extraction
   │
   ▼
30-Frame Sequence
   │
   ▼
LSTM Neural Network
   │
   ▼
Gesture Prediction
   │
   ▼
Real-Time Web Interface

```

## How It Works

1. Video Capture

The webcam continuously captures video frames using OpenCV.

2. Hand Landmark Detection

MediaPipe detects hand landmarks from each frame.

Each detected hand contains 21 landmark points with:

X coordinate
Y coordinate
Z coordinate

The system uses the wrist as the reference point to create normalized landmark features.

3. Sequence Formation

Instead of analyzing a single frame, the system collects a sequence of 30 consecutive frames.

This allows the model to learn both:

Hand position
Hand movement over time

4. Deep Learning Prediction

The extracted landmark sequence is passed to an LSTM neural network.

The model analyzes the temporal movement pattern and produces a gesture prediction with a confidence score.

5. Real-Time Display

The prediction is displayed directly on the live webcam interface.

Machine Learning Model

The project uses a Long Short-Term Memory (LSTM) neural network because sign gestures can involve movement across multiple frames.

For two hands, the feature representation is:

21 landmarks × 3 coordinates × 2 hands
= 126 features per frame

The model processes:

30 frames × 126 features

as one input sequence.

## Model Architecture

```text

Input Sequence
      ↓
LSTM (128 units)
      ↓
Dropout
      ↓
LSTM (64 units)
      ↓
Dropout
      ↓
Dense Layer
      ↓
Output Layer

```

## Model Performance

The trained prototype achieved:

96.25% test accuracy

on the held-out evaluation dataset.

The evaluation results demonstrate that the model can effectively learn the landmark-based movement patterns present in the collected dataset.

Test accuracy represents performance on the held-out dataset. Real-world performance can vary depending on lighting, camera position, hand orientation, background, and signing style.

## Current Scope

The current prototype demonstrates real-time recognition of a selected set of ASL gestures.

The architecture is designed to be extensible, allowing additional gesture classes and larger datasets to be incorporated into future versions.

The project focuses on gesture recognition, rather than claiming to provide complete ASL sentence translation.

## Project Structure

```text 
SignBridge/
│
├── dataset/
│   ├── hello.csv
│   ├── yes.csv
│   ├── no.csv
│   └── sorry.csv
│
├── model/
│   ├── labels.txt
│   └── sign_model.keras
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── APP.py
├── collect_data.py
├── train_model.py
├── hand_landmarker.task
├── requirements.txt
└── README.md

```

## Installation

1. Clone the repository
git clone <YOUR-GITHUB-REPOSITORY-URL>

Move into the project directory:

cd SignBridge
2. Create a virtual environment
python -m venv .venv
3. Activate the virtual environment
Windows
.venv\Scripts\activate
4. Install dependencies
pip install -r requirements.txt
Running the Application

Start the Flask application:

python APP.py

The application will start at:

http://127.0.0.1:5000

Open the address in a web browser and allow webcam access when prompted.

The system will process the live camera feed and display the predicted gesture and confidence score.

Data Collection

The collect_data.py script can be used to collect landmark-based gesture sequences.

The data collection pipeline is:

```text 

Webcam
   ↓
MediaPipe
   ↓
Hand Landmark Detection
   ↓
Feature Extraction
   ↓
Sequence Formation
   ↓
CSV Dataset

```

Each recorded sequence contains 30 frames of hand landmark information.

Training the Model

To train the model using the available datasets:

python train_model.py

The training pipeline:

Loads the gesture datasets.
Validates the landmark features.
Converts sequences into model-ready input.
Encodes gesture labels.
Splits the dataset into training and testing sets.
Trains the LSTM model.
Evaluates model performance.
Saves the trained model.

The trained model is saved as:

model/sign_model.keras

The gesture labels are saved as:

model/labels.txt
Web Application

The Flask application provides a browser-based interface for real-time recognition.

The application integrates:

```text

Flask
  │
  ├── Webcam Stream
  │
  ├── MediaPipe
  │
  ├── Trained LSTM Model
  │
  └── Real-Time Prediction

```

## Limitations

The current implementation is a prototype and has several limitations:

Recognition is currently limited to the gestures represented in the training dataset.
Performance may vary under different lighting conditions.
Camera position and hand orientation can affect recognition.
The model is trained on a limited dataset.
Different users may perform the same gesture differently.
The system focuses on isolated gesture recognition rather than continuous sentence-level translation.

## Future Improvements

Possible future improvements include:

Expanding the gesture vocabulary.
Using larger and more diverse datasets.
Supporting continuous sign-language sequences.
Improving model generalization across different users.
Improving robustness to different backgrounds and lighting conditions.
Adding text-to-speech functionality.
Adding multilingual output support.
Deploying the application as a cloud-based service.
Improving model performance with additional training data.

## Project Workflow

``` text 

Data Collection
      ↓
Data Preprocessing
      ↓
MediaPipe Landmark Extraction
      ↓
Sequence Generation
      ↓
LSTM Model Training
      ↓
Model Evaluation
      ↓
Flask Integration
      ↓
Real-Time Webcam Recognition

```