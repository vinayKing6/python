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

model=models.load_model('./reuters.h5')

test_label=test_labels[0]
test=vectorize_sequences((test_data[0],),10000)
print(test.shape)
results=model.predict(test)
print(results)
print(results.shape)
prediction=np.argmax(results)
print(f"{test_label} {prediction}")
