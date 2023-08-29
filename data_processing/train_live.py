import cv2
import mediapipe as mp
import data_processing.utils as u
import data_processing.write_csv as w

def load_labels():
    ges = {}
    with open('gesture_names.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            num, label = line.strip().split(': ')
            num = int(num)
            ges[num] = label
    return ges.keys(), ges.values()

def data_init():
    print("How many images per label?")
    num_im = int(input())
    # num_im = 20
    return num_im

def train_live(num_label, num_im):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0)
    i = 0
    j = 0
    with mp_hands.Hands(
            model_complexity=0,
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
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_list = u.calc_landmark_list(image, hand_landmarks)
                    pre_processed_landmark_list = u.pre_process_landmark(
                        landmark_list)

                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            final = cv2.flip(image, 1)
            cv2.imshow('MediaPipe Hands', final)
            cv2.putText(final, f'Label {num_label[j]}, Count: {i+1}', (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255))
            cv2.imshow('MediaPipe Hands', final)
            key = cv2.waitKey(200)
            if key == 27:
                break
            if key == ord('c'):
                w.write_csv(num_label[j], pre_processed_landmark_list)
                i += 1
            if i == num_im:
                j += 1
                i = 0
            if j == num_label[-1]:
                break

    cap.release()


if __name__ == '__main__':
    ges_num = []
    with open('landmark.csv', 'w') as f:
        f.truncate(0)
    key, _ = load_labels()
    for i in key:
        ges_num.append(i)
    num_im = data_init()
    train_live(ges_num, num_im)
