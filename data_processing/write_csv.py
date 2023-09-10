import pandas as pd

def write_csv(labels, landmarks_lst):
    csv_path = 'csv_file/landmark_9_9.csv'
    data = [[labels, *landmarks_lst]]
    df = pd.DataFrame(data)
    df.to_csv(csv_path, mode='a', header=False, index=False)


