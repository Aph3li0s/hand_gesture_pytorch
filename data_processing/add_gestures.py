def add_ges():
    gestures = {}
    print("Input gestures size: ")
    size = int(input())
    for i in range(size):
        gesture_name = input(f"Name of gesture {i+1}: ")
        gestures[i] = gesture_name

    return gestures

def save_ges(gestures):
    txt_dir = 'gesture_names2.txt'
    with open(txt_dir, 'w') as f:
        # f.write(f"0: No hand detected\n")
        for num, label in gestures.items():
            f.write(f"{num}: {label}\n")
        f.write(f"{num+1}: No hand detected\n")


if __name__ == '__main__':
    print(save_ges(add_ges()))
