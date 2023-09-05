import pandas as pd

def write_csv(labels, landmarks_lst):
    csv_path = 'csv_file/landmark_5_9_4.csv'
    data = [[labels, *landmarks_lst]]
    df = pd.DataFrame(data)
    df.to_csv(csv_path, mode='a', header=False, index=False)


if __name__ == '__main__':
    classes = [1, 2, 3]
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    write_csv(classes, data)
