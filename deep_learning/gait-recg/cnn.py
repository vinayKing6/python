import os

import matplotlib.pyplot as plt
from data_process import *
from network import *

#提取数据
directory='./dataset/Dataset I'
for classes in os.listdir(directory):
    path=os.path.join(directory,classes)
    classfier = len(os.listdir(os.path.join(path, 'train')))
    save_path = classes
    val_batch = 400
    epoch = 40
    loop = 1

    x_train, y_train = load_data(os.path.join(path, 'train'))
    print(x_train.shape, y_train.shape)

    # one hot
    y_train = one_hot(y_train, classfier)
    print(y_train.shape)
    print(y_train)

    # cnn
    cnn = CNN()
    cnn.build((6, 100, 1), classfier)

    # validation_data
    # mean=x_train.mean(axis=0)
    # std=x_train.std(axis=0)
    # x_train-=mean
    # x_train/=std

    x_val = x_train[:val_batch]
    y_val = y_train[:val_batch]
    partial_x_train = x_train[val_batch:]
    partial_y_train = y_train[val_batch:]

    print(x_val.shape)
    print(partial_x_train.shape)

    # run model
    history = cnn.run(epoch, partial_x_train, partial_y_train, x_val, y_val, save_path, loop=loop)
    model = cnn.get_model()

    # get test data
    x_test, y_test = load_data(os.path.join(path, 'test'), shuffle=False)
    print(x_test.shape, y_test.shape)

    # one_hot
    y_test = one_hot(y_test, classfier)
    print(y_test.shape)
    print(y_test)

    # x_test-=mean
    # x_test/=std
    print(model.evaluate(x_test, y_test))


