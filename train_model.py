import os
import csv
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical


DATASET_DIR = "dataset"
MODEL_DIR = "model"

GESTURES = [
    "hello",
    "yes",
    "no",
    "sorry"
]

SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 126


def load_dataset():

    X = []
    y = []

    print("\nLoading dataset...\n")

    for gesture in GESTURES:

        filename = os.path.join(
            DATASET_DIR,
            f"{gesture}.csv"
        )

        if not os.path.exists(filename):

            print(f"ERROR: {filename} not found.")
            continue

        count = 0

        with open(filename, "r") as file:

            reader = csv.reader(file)

            for row in reader:

                if len(row) < 2:
                    continue

                try:

                    label = row[-1]

                    values = [
                        float(value)
                        for value in row[:-1]
                    ]

                    expected_features = (
                        SEQUENCE_LENGTH *
                        FEATURES_PER_FRAME
                    )

                    if len(values) != expected_features:

                        print(
                            f"Skipping row in {gesture}: "
                            f"expected {expected_features} "
                            f"features, got {len(values)}"
                        )

                        continue

                    sequence = np.array(
                        values,
                        dtype=np.float32
                    )

                    sequence = sequence.reshape(
                        SEQUENCE_LENGTH,
                        FEATURES_PER_FRAME
                    )

                    X.append(sequence)
                    y.append(label)

                    count += 1

                except ValueError:

                    continue

        print(
            f"{gesture}: {count} sequences loaded"
        )

    return np.array(X), np.array(y)


def build_model(num_classes):

    model = Sequential([

        LSTM(
            128,
            return_sequences=True,
            input_shape=(
                SEQUENCE_LENGTH,
                FEATURES_PER_FRAME
            )
        ),

        Dropout(0.3),

        LSTM(64),

        Dropout(0.3),

        Dense(64, activation="relu"),

        Dropout(0.2),

        Dense(
            num_classes,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():

    print("=" * 55)
    print("          SIGNBRIDGE MODEL TRAINING")
    print("=" * 55)

    X, y = load_dataset()

    if len(X) == 0:

        print("\nERROR: No valid training data found.")
        return

    print("\nDataset shape:", X.shape)
    print("Labels:", y.shape)

    # Convert text labels to numbers.
    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)

    num_classes = len(
        label_encoder.classes_
    )

    print(
        "\nClasses:",
        list(label_encoder.classes_)
    )

    # Convert labels to one-hot encoding.
    y_categorical = to_categorical(
        y_encoded,
        num_classes=num_classes
    )

    # Split into training and testing data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_categorical,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Build model.
    model = build_model(num_classes)

    print("\nModel:")
    model.summary()

    # Train.
    print("\nStarting training...\n")

    history = model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )

    # Evaluate.
    print("\nEvaluating model...\n")

    predictions = model.predict(
        X_test,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    actual_classes = np.argmax(
        y_test,
        axis=1
    )

    accuracy = accuracy_score(
        actual_classes,
        predicted_classes
    )

    print(
        f"\nTest Accuracy: {accuracy * 100:.2f}%"
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            actual_classes,
            predicted_classes,
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )

    # Create model directory.
    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # Save trained model.
    model_path = os.path.join(
        MODEL_DIR,
        "sign_model.keras"
    )

    model.save(model_path)

    # Save class names.
    labels_path = os.path.join(
        MODEL_DIR,
        "labels.txt"
    )

    with open(
        labels_path,
        "w"
    ) as file:

        for label in label_encoder.classes_:

            file.write(
                label + "\n"
            )

    print("\n========================================")
    print("Training completed successfully!")
    print("========================================")

    print(
        f"\nModel saved to: {model_path}"
    )

    print(
        f"Labels saved to: {labels_path}"
    )


if __name__ == "__main__":
    main()