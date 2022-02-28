from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import os,shutil
import image_to_array

#files directory
base_dir='./Downloads/cats_and_dogs_small'

train_dir=os.path.join(base_dir,'train')
validation_dir=os.path.join(base_dir,'validation')
test_dir=os.path.join(base_dir,'test')

model=models.load_model('./cnn_2.h5')

test_data_generator=ImageDataGenerator(rescale=1./255)
test_generator=test_data_generator.flow_from_directory(test_dir,batch_size=1,target_size=(150,150),class_mode='binary')
test_generator.reset()
model.evaluate(test_generator)

test_image_path='./Downloads/cats_and_dogs_small/test/dogs/1538.jpg'
test_tensor=image_to_array.single_image(test_image_path,(150,150))

print(model.predict(test_tensor))
