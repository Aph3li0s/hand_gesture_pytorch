import cv2
import mediapipe as mp
import predict
import itertools

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def calc_landmark_list(landmarks):
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = landmark.x
        landmark_y = landmark.y
        landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y, landmark_z])
    temp_landmark_list = list(itertools.chain.from_iterable(landmark_point))
    return temp_landmark_list

def load_labels():
    ges = {}
    with open('data_processing/gesture_names2.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            num, label = line.strip().split(': ')
            num = int(num)
            ges[num] = label
    return ges.keys(), ges.values()

def combined_lst(landmarks_lst1, num, landmarks_lst2=None):
    zeros_list = [0] * 63
    if num == 0:
        final_lst = landmarks_lst1 + zeros_list
    elif num == 1:
        final_lst = zeros_list + landmarks_lst1
    else:
        final_lst = landmarks_lst1 + landmarks_lst2
    return final_lst

kpclf = predict.KeyPointClassifier()
key, value = load_labels()
gestures = dict(zip(key, value))

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():# 640x480
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        gesture_index = 8
        if results.multi_hand_landmarks and results.multi_handedness:
            left = []
            right = []
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                landmark_list = calc_landmark_list(hand_landmarks)

                if handedness.classification[0].label == "Left":
                    right.extend(landmark_list)
                    # print("R")
                elif handedness.classification[0].label == "Right":
                    left.extend(landmark_list)
                    # print("L")
            if len(results.multi_handedness) == 1:
                if handedness.classification[0].label == "Left":
                    new_lst = combined_lst(right, 1)
                if handedness.classification[0].label == "Right":
                    new_lst = combined_lst(left, 0)
            elif len(results.multi_handedness) == 2:
                new_lst = combined_lst(left, 2, right)  # Combine both left and right landmarks
            gesture_index = kpclf(new_lst)

        final = cv2.flip(image, 1)
        cv2.putText(final, gestures[gesture_index],
                    (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, 255)
        cv2.imshow('MediaPipe Hands', final)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
