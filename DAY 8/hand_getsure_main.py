import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=10,  # Allow detection of up to 10 hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame for natural interaction
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        hand_count = 0
        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Finger counting logic
                finger_tips = [4, 8, 12, 16, 20]
                finger_pips = [2, 6, 10, 14, 18]
                fingers = []

                # Thumb
                if hand_landmarks.landmark[finger_tips[0]].x > hand_landmarks.landmark[finger_pips[0]].x:
                    fingers.append(1)
                else:
                    fingers.append(0)
                # Other four fingers
                for tip, pip in zip(finger_tips[1:], finger_pips[1:]):
                    if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                finger_count = sum(fingers)

                # Display finger count for each hand
                cv2.putText(frame, f'Fingers: {finger_count}', 
                            (10, 80 + idx * 40), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (255, 255, 0), 2)
                print(f'Hand {idx+1}: {finger_count} fingers')

        # Display hand count on UI
        cv2.rectangle(frame, (0,0), (250,60), (0,0,0), -1)
        cv2.putText(frame, f'Hands: {hand_count}', (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

        cv2.imshow('Hand Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()