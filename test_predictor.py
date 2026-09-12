import cv2

from predictor import SignLanguagePredictor


predictor = SignLanguagePredictor()

camera = cv2.VideoCapture(0)

while True:

    ret, frame = camera.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    letter, confidence, results = predictor.predict(frame)

    frame = predictor.draw_landmarks(
        frame,
        results
    )

    if letter is not None:

        text = f"{letter} - {confidence * 100:.2f}%"

        cv2.putText(
            frame,
            text,
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

    cv2.imshow(
        "SignCare AI Prediction",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


camera.release()

cv2.destroyAllWindows()

predictor.close()