import cv2
import numpy as np
import mediapipe as mp
import pyautogui

# Initialize mediapipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Get screen size
screen_w, screen_h = pyautogui.size()

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get index finger tip coordinates
            x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
            y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)

            # Get thumb tip coordinates
            thumb_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * w)
            thumb_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * h)

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Map coordinates to screen size
            screen_x = np.interp(x, (0, w), (0, screen_w))
            screen_y = np.interp(y, (0, h), (0, screen_h))

            # Move mouse
            pyautogui.moveTo(screen_x, screen_y)

            # Visual feedback
            cv2.circle(frame, (x, y), 10, (255, 0, 255), cv2.FILLED)

            # Click functionality: if index tip and thumb tip are close, click
            distance = np.hypot(x - thumb_x, y - thumb_y)
            if distance < 40:  # Threshold for click, adjust as needed
                cv2.circle(frame, ((x + thumb_x)//2, (y + thumb_y)//2), 15, (0,255,0), cv2.FILLED)
                pyautogui.click()
                cv2.putText(frame, "Click", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Hand Cursor Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()