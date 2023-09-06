import cv2
import mediapipe as mp
import data_processing.write_csv as w
import itertools
import utils as u
def load_labels():
    ges = {}
    with open('gesture_names2.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            num, label = line.strip().split(': ')
            num = int(num)
            ges[num] = label
    return ges.keys(), ges.values()

def data_init():
    num_im = 10
    return num_im

def calc_landmark_list(landmarks):
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = landmark.x
        landmark_y = landmark.y
        landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y, landmark_z])
    temp_landmark_list = list(itertools.chain.from_iterable(landmark_point))
    return temp_landmark_list

def combined_lst(landmarks_lst1, num, landmarks_lst2=None):
    # Left first, then Right
    zeros_list = [0] * 63
    if num == 0:
        final_lst = landmarks_lst1 + zeros_list
    elif num == 1:
        final_lst = zeros_list + landmarks_lst1
    else:
        final_lst = landmarks_lst1 + landmarks_lst2
    return final_lst
num_class_lst = [1000, 800, 800, 700, 700, 700, 700, 500]
def train_live(num_label, num_im):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0)
    i = 0
    j = 3
    with mp_hands.Hands(
            model_complexity=0,
            max_num_hands=2,
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
                    elif handedness.classification[0].label == "Right":
                        left.extend(landmark_list)
                if len(results.multi_handedness) == 1:
                    if handedness.classification[0].label == "Left":
                        new_lst = combined_lst(right, 1)
                    if handedness.classification[0].label == "Right":
                        new_lst = combined_lst(left, 0)
                else:
                    new_lst = combined_lst(left, 2, right)  # Combine both left and right landmarks
            final = cv2.flip(image, 1)
            cv2.imshow('MediaPipe Hands', final)
            cv2.putText(final, f'Label {num_label[j]}, Count: {i}', (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, 255)
            cv2.imshow('MediaPipe Hands', final)
            key = cv2.waitKey(200)
            if key == 27:
                break
            if key == ord(' '):
                print(new_lst)
                w.write_csv(num_label[j], new_lst)
                i += 1
            if i == num_class_lst[j]:
                j += 1
                i = 0
            if j == num_label[-1]:
                break

    cap.release()


if __name__ == '__main__':
    ges_num = []
    with open('csv_file/landmark_5_9_4.csv', 'w') as f:
        f.truncate(0)
    key, _ = load_labels()
    for i in key:
        ges_num.append(i)
    num_im = data_init()
    train_live(ges_num, num_im)
