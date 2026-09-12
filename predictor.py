import cv2
import mediapipe as mp
import pickle
import os


class SignLanguagePredictor:

    def __init__(self, model_path="model/sign_language_model.p"):

        # -----------------------------
        # Load AI Model
        # -----------------------------

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        model_dict = pickle.load(
            open(model_path, "rb")
        )

        self.model = model_dict["model"]

        # -----------------------------
        # Initialize MediaPipe Hands
        # -----------------------------

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # -----------------------------
        # Drawing Utilities
        # -----------------------------

        self.mp_draw = mp.solutions.drawing_utils

    # =====================================================
    # Predict Hand Sign
    # =====================================================

    def predict(self, frame):

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Process image with MediaPipe
        results = self.hands.process(rgb_frame)

        # If no hand is detected
        if not results.multi_hand_landmarks:

            return None, 0.0, results

        # Get first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # -----------------------------
        # Wrist Normalization
        # -----------------------------

        wrist_x = hand_landmarks.landmark[0].x
        wrist_y = hand_landmarks.landmark[0].y
        wrist_z = hand_landmarks.landmark[0].z

        landmarks = []

        # Extract 21 landmarks
        for landmark in hand_landmarks.landmark:

            normalized_x = landmark.x - wrist_x
            normalized_y = landmark.y - wrist_y
            normalized_z = landmark.z - wrist_z

            landmarks.extend([
                normalized_x,
                normalized_y,
                normalized_z
            ])

        # -----------------------------
        # AI Prediction
        # -----------------------------

        prediction = self.model.predict(
            [landmarks]
        )

        predicted_letter = prediction[0]

        # -----------------------------
        # Confidence Score
        # -----------------------------

        confidence = 0.0

        if hasattr(
            self.model,
            "predict_proba"
        ):

            probabilities = self.model.predict_proba(
                [landmarks]
            )

            confidence = max(
                probabilities[0]
            )

        return (
            predicted_letter,
            confidence,
            results
        )

    # =====================================================
    # Draw Hand Landmarks
    # =====================================================

    def draw_landmarks(
        self,
        frame,
        results
    ):

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

        return frame

    # =====================================================
    # Release MediaPipe
    # =====================================================

    def close(self):

        self.hands.close()