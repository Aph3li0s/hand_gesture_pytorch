import pandas as pd

a = []
for i in range(66):
    if i % 3 != 0:
        a.append(i)
# df = pd.read_csv('landmark.csv', usecols=a)
# a = df.iloc[:, 1:]
# a.to_csv('test_z_2.csv')
df = pd.read_csv('landmark.csv', usecols=a)
# new_df = pd.DataFrame(df[a])