#回归问题：预测房价
from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np

(train_data,train_targets),(test_data,test_targets)=boston_housing.load_data()

#数据标准化
mean=train_data.mean(axis=0)
train_data-=mean
std=train_data.std(axis=0)
train_data/=std

test_data-=mean
test_data/=std

model=models.load_model('./boston_housing.h5')
test_mse,test_mae=model.evaluate(test_data,test_targets)
print(test_mae)
