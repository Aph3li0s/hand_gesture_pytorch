import mediapipe as mp


class LiveWebcam:
    def __int__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
    pass


if __name__ == '__main__':
    test = LiveWebcam()

