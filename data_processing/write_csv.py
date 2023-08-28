import pandas as pd

def write_csv(labels, landmarks_lst):
    csv_dir = 'landmark.csv'
    data = []
    for label, landmarks in zip(labels, landmarks_lst):
        data.append([label, *landmarks])
    print(data)
    df = pd.DataFrame(data)
    df.to_csv(csv_dir, index=False, header=False)


if __name__ == '__main__':
    classes = [1, 2, 3]
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    write_csv(classes, data)
