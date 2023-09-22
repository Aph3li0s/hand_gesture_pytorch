import pandas as pd

def write_csv(labels, landmarks_lst):
    csv_path = 'data_processing/csv_file/test.csv'
    data = [[labels, *landmarks_lst]]
    df = pd.DataFrame(data)
    df.to_csv(csv_path, mode='a', header=False, index=False)


