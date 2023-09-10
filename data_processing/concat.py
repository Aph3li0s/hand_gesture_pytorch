import pandas as pd

def add_rows():
    input_file = 'csv_file/landmark_8_9_1.csv'
    output_file = 'csv_file/a.csv'

    specific_number = 1
    df = pd.read_csv(input_file)
    filtered_df = df[df.iloc[:, 0] == specific_number]
    filtered_df.to_csv(output_file, mode='a', header=False, index=False)


def remove_rows():
    # file_path = 'csv_file/b.csv'
    # df = pd.read_csv(file_path)
    # df = df[df.iloc[:, 5] != 0]
    # df.reset_index(drop=True, inplace=True)
    # df.to_csv(file_path, index=False)
    with open('csv_file/b.csv', 'w') as f:
        f.truncate(0)

add_rows()
# remove_rows()