from data_process import *
from keras import models
from network import *
import os

directory = './dataset/Dataset III'
result = {}
# for classes in os.listdir(directory):
#     path = os.path.join(directory, classes)
classfiers = len(os.listdir(os.path.join(directory, 'test')))
model_path = 'III'
x_test, y_test = load_data(os.path.join(directory, 'test'), shuffle=False)
y_test = one_hot(y_test, classfiers)
print(x_test.shape, y_test.shape)
print(y_test)
model, acc = CNN.find_best_fit_model(x_test, y_test, model_path)

print(acc)

# test
test_index = 1400
print(np.array((x_test[test_index],)).shape)
test = model.predict(np.array((x_test[test_index],)))
print(test)
print(np.argmax(test), y_test[test_index, np.argmax(test)])
