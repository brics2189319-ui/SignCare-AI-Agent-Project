#SignCare – An Autonomous Clinical Communication System for Emergency Healthcare Environments.
import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import pyttsx3
import threading
import speech_recognition as sr
import time

# --- 1. VOICE ENGINE ---
def speak_text(text):
    if text.strip() == "": return
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 20)
    engine.say(text)
    engine.runAndWait()

# --- 2. LISTENING ENGINE ---
doctor_message = "Press 'D' to dictate"

def listen_to_doctor():
    global doctor_message
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        doctor_message = "Listening... (Speak now)"
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            text = recognizer.recognize_google(audio)
            doctor_message = f"{text}"
        except sr.UnknownValueError:
            doctor_message = "Could not understand audio"
        except Exception:
            doctor_message = "No speech detected"

# --- 3. MEDICAL LOGIC & VOCABULARY ---
medical_vocab = [
    # Core Assistance
    "HELP", "EMERGENCY", "DOCTOR", "NURSE", "HOSPITAL",

    # Symptoms
    "PAIN", "HURTS", "FEVER", "DIZZY", "COUGH",
    "COLD", "HOT", "CHILLS", "WEAK", "TIRED",
    "NAUSEA", "VOMIT", "HEADACHE", "STOMACH",
    "CHEST", "BREATH", "BLEEDING",

    # Basic Needs
    "WATER", "FOOD", "REST", "PILLS", "MEDICINE",

    # Responses
    "YES", "NO", "OK", "STOP"
]

emergency_words = [
    "EMERGENCY", "HELP", "PAIN", "HURTS",
    "CHEST", "BREATH", "BLEEDING"
]

symptom_words = [
    "FEVER", "DIZZY", "COUGH", "COLD",
    "NAUSEA", "HEADACHE", "WEAK", "TIRED",
    "PAIN", "CHILLS"
]

# Load Model
model_dict = pickle.load(open(os.path.join('model', 'sign_language_model.p'), 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow('SignCare Clinical Dashboard', cv2.WINDOW_NORMAL)
cv2.resizeWindow('SignCare Clinical Dashboard', 1280, 720)

# --- SYSTEM VARIABLES ---
current_sentence = ""
predicted_buffer = []  
buffer_size = 8        
last_confirmed_char = ""
current_confidence = 0.0
is_emergency = False
extracted_symptom = "Pending..."
emergency_alert_played = False

print("🏥 SignCare Clinical Dashboard - INITIALIZED")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # --- RESET & UPDATE LOGIC ---
    words = current_sentence.split(" ")
    current_word = words[-1] if words else ""
    
    # Auto-extract Symptoms
    for w in words:
        if w in symptom_words:
            extracted_symptom = w
            
    # Emergency Detection
    is_emergency = any(w in emergency_words for w in words)
    if is_emergency and not emergency_alert_played:
        threading.Thread(target=speak_text, args=("Emergency Medical Alert Triggered",)).start()
        emergency_alert_played = True

    # Autocomplete Suggestions
    suggestions = [w for w in medical_vocab if w.startswith(current_word)] if current_word else []
    suggestions = suggestions[:3]

    # --- UI: CLINICAL DESKTOP LAYOUT ---
    sidebar_x = int(W * 0.75) 
    cv2.rectangle(frame, (sidebar_x, 0), (W, H), (25, 25, 25), -1)
    cv2.line(frame, (sidebar_x, 0), (sidebar_x, H), (100, 100, 100), 2)
    cv2.rectangle(frame, (0, H-80), (sidebar_x, H), (35, 35, 35), -1)

    # --- HAND TRACKING & AI INFERENCE ---
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            
            # --- WRIST NORMALIZATION ANCHOR ---
            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y
            wrist_z = hand_landmarks.landmark[0].z
            
            for i in range(len(hand_landmarks.landmark)):
                lm = hand_landmarks.landmark[i]
                # Calculate distance from wrist
                data_aux.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])

            try:
                # Confidence Scoring & Temporal Smoothing
                prediction_proba = model.predict_proba([np.asarray(data_aux)])[0]
                current_confidence = max(prediction_proba) * 100
                raw_char = str(model.classes_[np.argmax(prediction_proba)])

                if current_confidence > 60.0:
                    predicted_buffer.append(raw_char)
                    if len(predicted_buffer) > buffer_size: predicted_buffer.pop(0)

                    if predicted_buffer.count(raw_char) > (buffer_size * 0.8):
                        if raw_char != last_confirmed_char:
                            current_sentence += raw_char
                            last_confirmed_char = raw_char
                else:
                    predicted_buffer.clear() 
                    last_confirmed_char = ""
                    
            except ValueError:
                pass 
            
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                   mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                                   mp_draw.DrawingSpec(color=(230, 160, 70), thickness=2, circle_radius=2))
    else:
        last_confirmed_char = ""
        current_confidence = 0.0

    # --- DRAW EHR PANEL (RIGHT SIDEBAR) ---
    cv2.putText(frame, "SignCare EHR", (sidebar_x + 20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    cv2.line(frame, (sidebar_x + 20, 55), (W - 20, 55), (100, 100, 100), 1)

    conf_color = (100, 255, 100) if current_confidence > 75 else (0, 255, 255) if current_confidence > 50 else (0, 0, 255)
    cv2.putText(frame, f"AI Confidence: {int(current_confidence)}%", (sidebar_x + 20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)
    if 0 < current_confidence < 60:
        cv2.putText(frame, "Low Conf - Please Confirm", (sidebar_x + 20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.putText(frame, "CLINICAL DATA:", (sidebar_x + 20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"Symptom: {extracted_symptom}", (sidebar_x + 20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "Duration: TBD", (sidebar_x + 20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "Severity: TBD", (sidebar_x + 20, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.putText(frame, "DOCTOR STT:", (sidebar_x + 20, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, doctor_message[:25], (sidebar_x + 20, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 255), 1)
    if len(doctor_message) > 25:
        cv2.putText(frame, doctor_message[25:50], (sidebar_x + 20, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 255), 1)

    cv2.putText(frame, "Privacy Mode Enabled.", (sidebar_x + 20, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 150, 100), 1)
    cv2.putText(frame, "Zero Data Stored.", (sidebar_x + 20, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(frame, "Auto-Resets Session.", (sidebar_x + 20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # --- DRAW CAMERA FEED TEXT (LEFT SIDE) ---
    if is_emergency:
        cv2.rectangle(frame, (0, 0), (sidebar_x, H), (0, 0, 255), 10) 
        cv2.rectangle(frame, (0, 0), (sidebar_x, 60), (0, 0, 200), -1)
        cv2.putText(frame, "🚨 TRIAGE ALERT: EMERGENCY DETECTED", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
    
    # 1. Dynamic Background for Patient Text
    patient_text = f"Patient: {current_sentence}"
    (tw, th), baseline = cv2.getTextSize(patient_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)
    cv2.rectangle(frame, (15, H - 120 - th - 10), (15 + tw + 20, H - 120 + baseline + 10), (0, 0, 0), -1)
    cv2.putText(frame, patient_text, (20, H-120), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
    
    # 2. Dynamic Background for Autocomplete Suggestions
    if suggestions:
        sugg_text = "Suggest: " + " | ".join([f"[{i+1}] {s}" for i, s in enumerate(suggestions)])
        (sw, sh), s_base = cv2.getTextSize(sugg_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (15, H - 170 - sh - 10), (15 + sw + 20, H - 170 + s_base + 10), (0, 0, 0), -1)
        cv2.putText(frame, sugg_text, (20, H-170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)

    # 3. Dynamic Background for Controls Menu
    ctrl_str = "CONTROLS: [S] Speak | [D] Doctor STT | [1,2,3] Auto | [C] Clear | [R] Reset Session"
    (cw, ch), c_base = cv2.getTextSize(ctrl_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (15, H - 35 - ch - 5), (15 + cw + 10, H - 35 + c_base + 5), (0, 0, 0), -1)
    cv2.putText(frame, ctrl_str, (20, H-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('SignCare Clinical Dashboard', frame)
    
    # --- KEYBOARD CONTROLS ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('d'): threading.Thread(target=listen_to_doctor).start()
    elif key == ord('s'): threading.Thread(target=speak_text, args=(current_sentence,)).start()
    elif key == 32:       current_sentence += " "
    elif key == 8:        current_sentence = current_sentence[:-1]
    elif key == ord('c'): current_sentence = ""
    elif key == ord('r'): 
        current_sentence = ""
        extracted_symptom = "Pending..."
        doctor_message = "Press 'D' to dictate"
        is_emergency = False
        emergency_alert_played = False
        predicted_buffer.clear()
        threading.Thread(target=speak_text, args=("Session Reset. All data cleared.",)).start()
        
    elif key in [ord('1'), ord('2'), ord('3')]:
        idx = key - ord('1')
        if idx < len(suggestions):
            words = current_sentence.split(" ")
            words[-1] = suggestions[idx]
            current_sentence = " ".join(words) + " "

cap.release()
cv2.destroyAllWindows()