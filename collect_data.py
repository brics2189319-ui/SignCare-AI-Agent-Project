import cv2
import mediapipe as mp
import csv
import os
import time

# --- 1. SET UP MEDIAPIPIE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# --- 2. SET UP DATA STORAGE ---
csv_path = os.path.join('Data', 'alphabet_data.csv')
if not os.path.exists('Data'):
    os.makedirs('Data')

# Create CSV header only if the file doesn't exist yet
if not os.path.exists(csv_path):
    with open(csv_path, mode='w', newline='') as f:
        header = [f'p_{i}_{c}' for i in range(21) for c in ['x', 'y', 'z']] + ['label']
        csv.writer(f).writerow(header)

# --- 3. INITIALIZE CAMERA (STANDARD RESOLUTION) ---
def initialize_camera():
    for index in [0, 1]: 
        print(f"⌛ Attempting to open camera index {index}...")
        
        # CAP_DSHOW kept for crash-prevention, but NO HD resolution forced
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        time.sleep(2) 
        
        if cap.isOpened():
            for _ in range(5):
                ret, frame = cap.read()
            if ret:
                print(f"✅ Camera Index {index} is working safely!")
                return cap
            cap.release()
    return None

cap = initialize_camera()

if cap is None:
    print("❌ CRITICAL ERROR: Could not open any camera.")
    exit()

# --- 4. START COLLECTION PROCESS ---
print("\n🏥 SignCare Data Collection System")
target_sign = input("Which letter/sign are you recording? (e.g., A, B, C...): ").upper()

# --- LIVE PREVIEW COUNTDOWN ---
print(f"Look at the window! Get your '{target_sign}' hand sign ready.")
start_time = time.time()
while time.time() - start_time < 5: 
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

    countdown = int(5 - (time.time() - start_time)) + 1
    
    # Text scaled down to fit standard SD resolution
    cv2.putText(frame, f"GET READY: {countdown}", (W//2 - 150, H//2), 
                cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)
    cv2.imshow("SignCare Data Collection", frame)
    cv2.setWindowProperty("SignCare Data Collection", cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(1)

# --- AUTOMATIC DATA RECORDING ---
print(f"🔴 RECORDING '{target_sign}' NOW...")
count = 0
max_samples = 100 

while count < max_samples:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            landmarks = []
            mirrored_landmarks = []
            
            # --- WRIST NORMALIZATION ANCHOR ---
            wrist_x = hand_lms.landmark[0].x
            wrist_y = hand_lms.landmark[0].y
            wrist_z = hand_lms.landmark[0].z
            
            for lm in hand_lms.landmark:
                # 1. Normalized Right Hand (Distance from wrist)
                norm_x = lm.x - wrist_x
                norm_y = lm.y - wrist_y
                norm_z = lm.z - wrist_z
                landmarks.extend([norm_x, norm_y, norm_z])
                
                # 2. Normalized Left Hand
                mirrored_landmarks.extend([-norm_x, norm_y, norm_z])
                
            landmarks.append(target_sign)
            mirrored_landmarks.append(target_sign)
            
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(landmarks)
                writer.writerow(mirrored_landmarks)
            
            count += 1
            if count % 10 == 0:
                print(f"Captured: {count}/{max_samples}")

    # Text scaled down to fit standard SD resolution
    cv2.putText(frame, f"COLLECTING {target_sign}: {count}/{max_samples}", (15, 30), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("SignCare Data Collection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ SUCCESS: Finished recording {target_sign}!")