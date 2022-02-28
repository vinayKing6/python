import csv
import os
import numpy as np


# api

# 提取数据
def load_data(path, size=100, shuffle=True):
    X = []
    Y = []
    for i in range(len(os.listdir(path))):
        temp = []
        rows = []
        fname = os.path.join(path, '{}.csv'.format(i + 1))
        with open(fname) as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # header
            for row in csv_reader:
                rows.append(row)
        for row in rows:
            X.append([float(x) for x in row])
            temp.append([float(x) for x in row])
        temp = np.array(temp)
        temp = temp.reshape(-1, size, 6, 1)
        for n in range(temp.shape[0]):
            Y.append(i)
    X = np.array(X)
    X = X.reshape(-1, size, 6, 1)
    X = np.transpose(X, (0, 2, 1, 3))
    Y = np.array(Y)

    if shuffle:
        state = np.random.get_state()
        np.random.set_state(state)
        np.random.shuffle(X)
        np.random.set_state(state)
        np.random.shuffle(Y)

    return X, Y


# one hot
def one_hot(data, capacity):
    result = np.zeros((len(data), capacity))
    for i, pos in enumerate(data):
        result[i, pos] = 1.
    return result


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
