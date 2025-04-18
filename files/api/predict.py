# --- START OF FILE files/api/predict.py ---

import tensorflow as tf
import os, sys
import numpy as np
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
# Use static_image_mode=True for processing single images
# Set max_num_hands=1 if you only want to detect/process one hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

EXPECTED_LANDMARKS = 21
EXPECTED_FEATURES = EXPECTED_LANDMARKS * 3 # x, y, z

def get_marks(image_path):
    """
    Processes an image file to extract hand landmarks for the first detected hand.

    Args:
        image_path (str): The path to the image file.

    Returns:
        np.ndarray: A numpy array of shape (1, 63) containing the flattened
                    x, y, z coordinates of the hand landmarks, normalized relative
                    to the first landmark (wrist). Returns None if no hand is
                    detected or landmarks are incomplete.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None

    # Load the image
    image = cv2.imread(image_path) # BUG 5 FIX: Use the variable image_path
    if image is None:
        print(f"Error: Could not read image file at {image_path}")
        return None

    # Convert the image to RGB (MediaPipe requires RGB format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to find hands
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        # Get landmarks for the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # Check if the expected number of landmarks are detected
        if len(hand_landmarks.landmark) != EXPECTED_LANDMARKS:
            print(f"Warning: Detected {len(hand_landmarks.landmark)} landmarks, expected {EXPECTED_LANDMARKS}.")
            # Depending on the model's robustness, you might still proceed or return None
            # return None # Stricter approach

        landmarks_list = []
        # Extract x, y, z coordinates
        for landmark in hand_landmarks.landmark:
            landmarks_list.append([landmark.x, landmark.y, landmark.z])

        # Convert to NumPy array
        landmarks_array = np.array(landmarks_list) # Shape: (21, 3)

        # Optional: Normalize landmarks relative to the wrist (landmark 0)
        # This makes the model potentially more robust to hand position/scale
        # base_x, base_y, base_z = landmarks_array[0]
        # normalized_landmarks = landmarks_array - [base_x, base_y, base_z]
        # Flatten the array
        # flattened_landmarks = normalized_landmarks.flatten()

        # Flatten the array directly if normalization wasn't part of training
        flattened_landmarks = landmarks_array.flatten() # Shape: (63,)

        # Reshape for model prediction (usually expects batch dimension)
        reshaped_landmarks = flattened_landmarks.reshape(1, EXPECTED_FEATURES) # Shape: (1, 63)

        return reshaped_landmarks

    # No hands detected
    return None

def _predict(landmark_data, loaded_model):
    """
    Predicts the ASL letter based on the provided landmark data.

    Args:
        landmark_data (np.ndarray or None): The landmark data array (e.g., shape (1, 63))
                                            or None if no valid landmarks were found.
        loaded_model: The loaded TensorFlow/Keras model.

    Returns:
        str: The predicted ASL letter (A-Z), or an empty string if prediction fails
             or landmark_data is None.
    """
    # REFACTOR 2: Handle None input and simplify for single prediction
    if landmark_data is None:
        print("Prediction skipped: No landmark data provided.")
        return "" # Return empty string or specific indicator

    if landmark_data.shape != (1, EXPECTED_FEATURES):
         print(f"Prediction skipped: Incorrect landmark data shape {landmark_data.shape}, expected (1, {EXPECTED_FEATURES}).")
         return ""

    try:
        # Make prediction
        prediction = loaded_model.predict(landmark_data) # prediction shape: (1, num_classes)

        # Check if prediction is valid
        if prediction is None or prediction.shape[0] < 1:
             print("Prediction failed: Model returned invalid output.")
             return ""

        # Get the index of the highest probability class for the first (only) sample
        index = np.argmax(prediction[0])

        # Convert index to character (assuming classes 0-25 map to A-Z)
        # Adjust the offset if your model's classes map differently
        predicted_char = chr(index + 65)

        return predicted_char

    except Exception as e:
        print(f"Error during prediction: {e}")
        return ""


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    image_file_path = sys.argv[1]

    # Provide the path to the directory where the model is saved
    # Assuming running from project root
    model_path_local = os.path.join(os.curdir, "files", "model", "asl_model.keras")

    if not os.path.exists(model_path_local):
         print(f"Error: Model file not found at {model_path_local}")
         sys.exit(1)

    # Load the saved model
    try:
        loaded_model_local = tf.keras.models.load_model(model_path_local, compile=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Get landmarks from the image
    landmarks = get_marks(image_file_path)

    if landmarks is not None:
        # Make prediction using the landmarks
        prediction_result = _predict(landmarks, loaded_model_local)
        print(f"Prediction for {image_file_path}: {prediction_result}")
    else:
        print(f"Could not get landmarks from {image_file_path}. No prediction made.")

# --- END OF FILE files/api/predict.py ---