from keras import models
from keras.datasets import imdb
import numpy as np

model=models.load_model('./imdb.h5')

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

#predict=model.predict(x_test)
#print(predict)

test='i love this movie because the actors performed just great'.split(' ')
test_code=[]
for word in test:
    test_code.append(word_index.get(word))
print(test_code)
test=vectorize_sequences((test_code,),10000)
print(model.predict(test))
