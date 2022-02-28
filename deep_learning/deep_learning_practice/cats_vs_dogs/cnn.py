from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import os,shutil

#files directory
base_dir='./Downloads/cats_and_dogs_small'

train_dir=os.path.join(base_dir,'train')
validation_dir=os.path.join(base_dir,'validation')
test_dir=os.path.join(base_dir,'test')

model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
#add dropout
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['acc'])

#load images from directory
train_data_generator=ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

test_data_generator=ImageDataGenerator(rescale=1./255)

train_generator=train_data_generator.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode='binary')
test_generator=test_data_generator.flow_from_directory(validation_dir,target_size=(150,150),batch_size=20,class_mode='binary')

#train model
history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=30,validation_data=test_generator,validation_steps=50)
model.save('./cnn_2.h5')

print(history.history['acc'])


