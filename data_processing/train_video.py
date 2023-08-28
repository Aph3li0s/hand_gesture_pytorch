import cv2
import mediapipe as mp
import data_processing.utils as u
import write_csv as w

def load_labels():
    ges = {}
    with open('gesture_names.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            num, label = line.strip().split(': ')
            num = int(num)
            ges[num] = label
    return ges.keys(), ges.values()


def train_video(num_label):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            number = cv2.waitKey(20) - 48
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks and number in num_label[1:]:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_list = u.calc_landmark_list(image, hand_landmarks)
                    pre_processed_landmark_list = u.pre_process_landmark(
                        landmark_list)
                    w.write_csv(number, pre_processed_landmark_list)
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            final = cv2.flip(image, 1)
            if number == -1:
                text = "Press key for gesture number"
            else:
                if number == -49:
                    text = "None"
                text = "Gesture: {}".format(number)
            cv2.putText(final, text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255))

            cv2.imshow('MediaPipe Hands', final)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


if __name__ == '__main__':
    ges_num = []
    with open('landmark.csv', 'w') as f:
        f.truncate(0)
    key, _ = load_labels()
    for i in key:
        ges_num.append(i)
    train_video(ges_num)