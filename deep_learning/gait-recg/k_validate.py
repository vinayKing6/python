import numpy as np
import matplotlib.pyplot as plt
from network import *
from data_process import *
import os

epoch_num = 50
train_path = './dataset/Dataset II/train'
classfier = len(os.listdir(train_path))
test_path = './dataset/Dataset II/test'
k = 4
all_scores = []

x_train, y_train = load_data(train_path)
x_test, y_test = load_data(test_path, shuffle=False)
y_train = one_hot(y_train, classfier)
y_test = one_hot(y_test, classfier)

# k-validation
val_batch_size = x_train.shape[0] // k

for i in range(k):
    print('process fold #', i)
    x_val = x_train[i * val_batch_size:(i + 1) * val_batch_size]
    y_val = y_train[i * val_batch_size:(i + 1) * val_batch_size]
    partial_x_train = np.concatenate([x_train[:i * val_batch_size], x_train[(i + 1) * val_batch_size:]], axis=0)
    partial_y_train = np.concatenate([y_train[:i * val_batch_size], y_train[(i + 1) * val_batch_size:]], axis=0)

    model = CNN()
    model.build((6, 100, 1), classfier)
    history = model.run(epoch_num, partial_x_train, partial_y_train, x_val, y_val, 'II', is_draw=False)
    all_scores.append(history[0].history['val_loss'])

average_val_loss = [np.mean([x[i] for x in all_scores]) for i in range(epoch_num)]
print(average_val_loss)

average_val_loss = smooth_curve(average_val_loss[35:])

plt.plot(range(1, len(average_val_loss) + 1), average_val_loss)
plt.show()
