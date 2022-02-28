import numpy as np

samples=['The cat sat one the mat.','The dog ate my homework.']

#one_hot by single word

def single_word_one_hot(samples):
    token_index={}

    max_length=0

    for sample in samples:
        for i,word in enumerate(sample.split()):
            if word not in token_index:
                token_index[word]=len(token_index)+1
            if (i+1)>max_length:
                max_length=i+1

    results=np.zeros(shape=(len(samples),max_length,max(token_index.values())+1))

    for i,sample in enumerate(samples):
        for j,word in list(enumerate(sample.split()))[:max_length]:
            index=token_index.get(word)
            results[i,j,index]=1.

    return results

print(single_word_one_hot(samples))
