from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt

(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)

#解码
word_index=imdb.get_word_index()
reverse_word_index=dict([(value,key) for (key,value) in word_index.items()])

#matrixrize by one hot 
def vectorize_sequences(sequences,dimension):
    results=np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1.
    return results

#准备数据
#数据
x_train=vectorize_sequences(train_data,10000)
x_test=vectorize_sequences(test_data,10000)
#标签（结果）
y_train=np.asarray(train_labels).astype('float32')
y_test=np.asarray(test_labels).astype('float32')

#准备模型
model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

#配置优化器、损失函数
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

#训练
model.fit(x_train,y_train,epochs=4,batch_size=512)
model.save('imdb.h5')

