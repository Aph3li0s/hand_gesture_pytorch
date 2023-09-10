import cv2
import mediapipe as mp
import data_processing.write_csv as w
import utils as u
import time

def train_live(num_label, num_im):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    timer = 3
    cap = cv2.VideoCapture(0)
    i = 0
    j = 0
    new_lst = []
    with mp_hands.Hands(
            model_complexity=0,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            final = cv2.flip(image, 1)
            cv2.imshow('MediaPipe Hands', final)
            cv2.putText(final, f'Label {num_label[j]}, Count: {i}', (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, 255)
            cv2.imshow('MediaPipe Hands', final)
            key = cv2.waitKey(1) & 0xff
            if key == 27:
                break
            if key == ord(' '):
                prev = time.time()
                while timer >= 0:
                    ret, img = cap.read()
                    final2 = cv2.flip(img, 1)
                    cv2.imshow('MediaPipe Hands', final2)
                    cv2.putText(final, f'Label {num_label[j]}, Count: {i}', (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, 255)
                    cv2.imshow('MediaPipe Hands', final2)
                    cv2.waitKey(10)
                    cur = time.time()
                    if cur - prev >= 1:
                        prev = cur
                        timer = timer - 1
                        print(timer)
                else:
                    for i in range(num_im):
                        success, image = cap.read()
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
                                landmark_list = u.calc_landmark_list(hand_landmarks)

                                if handedness.classification[0].label == "Left":
                                    right.extend(landmark_list)
                                elif handedness.classification[0].label == "Right":
                                    left.extend(landmark_list)
                            if len(results.multi_handedness) == 1:
                                if handedness.classification[0].label == "Left":
                                    new_lst = u.combined_lst(right, 1)
                                if handedness.classification[0].label == "Right":
                                    new_lst = u.combined_lst(left, 0)
                            else:
                                new_lst = u.combined_lst(left, 2, right)  # Combine both left and right landmarks
                        final = cv2.flip(image, 1)
                        cv2.imshow('MediaPipe Hands', final)
                        cv2.putText(final, f'Label {num_label[j]}, Count: {i}', (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, 255)
                        cv2.imshow('MediaPipe Hands', final)
                        cv2.waitKey(50)
                        print(new_lst)
                        w.write_csv(num_label[j], new_lst)
                    timer = 3
                    j += 1
                    key = cv2.waitKey(1) & 0xff
                    if key == 27:
                        break

    cap.release()

if __name__ == '__main__':
    ges_num = []
    # with open('csv_file/landmark_8_9_2.csv', 'w') as f:
    #     f.truncate(0)

    key, _ = u.load_labels()
    for i in key:
        ges_num.append(i)

    num_im = u.data_init()
    train_live(ges_num, num_im)

