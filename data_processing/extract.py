import pandas as pd

df = pd.read_csv('csv_file/output_file.csv', header=None)
x, y, z = [], [], []
num_columns = len(df.columns)
selected_columns = []

for col_index in range(num_columns):
    if col_index % 2 == 0:
        selected_columns.append(df.iloc[:, col_index])

selected_df = pd.concat(selected_columns, axis=1)

selected_df.to_csv('test.csv', index=False, header=False)
