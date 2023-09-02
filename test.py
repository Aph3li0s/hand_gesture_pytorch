import cv2
import mediapipe as mp
import predict
import data_processing.utils as u

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def load_labels():
    ges = {}
    with open('data_processing/gesture_names.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            num, label = line.strip().split(': ')
            num = int(num)
            ges[num] = label
    return ges.keys(), ges.values()


kpclf = predict.KeyPointClassifier()
key, value = load_labels()
gestures = dict(zip(key, value))

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        model_complexity=0,
        # max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gesture_index = 5
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = u.calc_landmark_list(image, hand_landmarks)
                keypoints = u.pre_process_landmark(landmark_list)
                gesture_index = kpclf(keypoints)

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        final = cv2.flip(image, 1)
        cv2.putText(final, gestures[gesture_index],
                    (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, 255)
        cv2.imshow('MediaPipe Hands', final)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
