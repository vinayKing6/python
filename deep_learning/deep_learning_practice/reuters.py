from keras.datasets import reuters
from keras import models
from keras import layers

import numpy as np

(train_data,train_labels),(test_data,test_labels)=reuters.load_data(num_words=10000)

#解码
word_index=reuters.get_word_index()
reverse_word_index=dict([(value,key) for (key,value) in word_index.items()])
first_news=' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])

#one hot train_data
def vectorize_sequences(sequences,dimension):
    results=np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1.
    return results

x_train=vectorize_sequences(train_data,10000)
x_test=vectorize_sequences(test_data,10000)
y_train=vectorize_sequences(train_labels,46)
y_test=vectorize_sequences(test_labels,46)

#构建模型
model=models.Sequential()
model.add(layers.Dense(128,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#留出验证集
x_val=x_train[:1000]
partial_x_train=x_train[1000:]
y_val=y_train[:1000]
partial_y_train=y_train[1000:]

model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))
results=model.evaluate(x_test,y_test)
print(results)
model.save('reuters.h5')
