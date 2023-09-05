import pandas as pd

input_file = 'csv_file/landmark_5_9_3.csv'
output_file = 'csv_file/output_file.csv'

specific_number = 2
df = pd.read_csv(input_file)
filtered_df = df[df.iloc[:, 0] == specific_number]
filtered_df.to_csv(output_file, mode='a', header=False, index=False)
