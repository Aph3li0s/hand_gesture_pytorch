import itertools

def load_labels():
    ges = {}
    with open('data_processing/gesture_names2.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            num, label = line.strip().split(': ')
            num = int(num)
            ges[num] = label
    return ges.keys(), ges.values()

def data_init():
    num_im = 1000
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
