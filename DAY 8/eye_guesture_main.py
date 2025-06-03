import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Screen size
screen_width, screen_height = pyautogui.size()

# For smoothing cursor movement
prev_x, prev_y = 0, 0
smoothing_factor = 0.7

# Click debounce timer
click_cooldown = 1.0  # seconds
last_click_time = 0

# Helper function to convert normalized landmarks to pixel coordinates
def landmark_to_point(landmark, frame_width, frame_height):
    return int(landmark.x * frame_width), int(landmark.y * frame_height)

# Calculate Eye Aspect Ratio (EAR) for blink detection
def eye_aspect_ratio(landmarks, eye_indices, frame_width, frame_height):
    # vertical distances
    p1 = landmarks[eye_indices[1]]
    p2 = landmarks[eye_indices[5]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[4]]
    # horizontal distance
    p0 = landmarks[eye_indices[0]]
    p5 = landmarks[eye_indices[3]]

    def dist(a, b):
        x1, y1 = landmark_to_point(a, frame_width, frame_height)
        x2, y2 = landmark_to_point(b, frame_width, frame_height)
        return np.linalg.norm([x2 - x1, y2 - y1])

    vertical_1 = dist(p1, p2)
    vertical_2 = dist(p3, p4)
    horizontal = dist(p0, p5)

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

# Right eye landmark indices (from MediaPipe Face Mesh)
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    frame_height, frame_width, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Use left iris landmark for cursor control
            left_iris_landmark = landmarks[474]
            x, y = landmark_to_point(left_iris_landmark, frame_width, frame_height)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            screen_x = np.interp(x, (0, frame_width), (0, screen_width))
            screen_y = np.interp(y, (0, frame_height), (0, screen_height))

            smooth_x = int(prev_x * smoothing_factor + screen_x * (1 - smoothing_factor))
            smooth_y = int(prev_y * smoothing_factor + screen_y * (1 - smoothing_factor))

            pyautogui.moveTo(smooth_x, smooth_y, duration=0.01)

            prev_x, prev_y = smooth_x, smooth_y

            # Calculate EAR for right eye
            ear = eye_aspect_ratio(landmarks, RIGHT_EYE_INDICES, frame_width, frame_height)

            # Draw EAR value on frame
            cv2.putText(frame, f'EAR: {ear:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # Blink detection threshold (adjust if needed)
            EAR_THRESHOLD = 0.25

            current_time = time.time()
            if ear < EAR_THRESHOLD and (current_time - last_click_time) > click_cooldown:
                pyautogui.click()
                last_click_time = current_time
                cv2.putText(frame, 'Click!', (screen_width//2, screen_height//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

    cv2.imshow("Eye Gesture Cursor Control with Click", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
