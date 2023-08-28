import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
import data_processing.utils as u

def load_labels():
    ges = {}
    with open('gesture_names.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            num, label = line.strip().split(': ')
            ges[num] = label
    return ges


def init_mp():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands


def train_video():
    pass


if __name__ == '__main__':
    gestures = load_labels()
    print(gestures)