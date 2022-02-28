import os,shutil

original_dataset_dir='./Downloads/PetImages'

base_dir='./Downloads/cats_and_dogs_small'
os.mkdir(base_dir)

train_dir=os.path.join(base_dir,'train')
os.mkdir(train_dir)
validation_dir=os.path.join(base_dir,'validation')
os.mkdir(validation_dir)
test_dir=os.path.join(base_dir,'test')
os.mkdir(test_dir)

#分类数据集
train_cats_dir=os.path.join(train_dir,'cats')
os.mkdir(train_cats_dir)
train_dogs_dir=os.path.join(train_dir,'dogs')
os.mkdir(train_dogs_dir)

validation_cats_dir=os.path.join(validation_dir,'cats')
os.mkdir(validation_cats_dir)
validation_dogs_dir=os.path.join(validation_dir,'dogs')
os.mkdir(validation_dogs_dir)

test_cats_dir=os.path.join(test_dir,'cats')
os.mkdir(test_cats_dir)
test_dogs_dir=os.path.join(test_dir,'dogs')
os.mkdir(test_dogs_dir)

#复制图像到指定文件夹

cats_src=os.path.join(original_dataset_dir,'Cat')
dogs_src=os.path.join(original_dataset_dir,'Dog')


#train_dir 1000 images for cats/dogs
fnames=['{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    cat_img=os.path.join(cats_src,fname)
    dog_img=os.path.join(dogs_src,fname)
    cat_cpy=os.path.join(train_cats_dir,fname)
    dog_cpy=os.path.join(train_dogs_dir,fname)
    shutil.copyfile(cat_img,cat_cpy)
    shutil.copyfile(dog_img,dog_cpy)

#validation_dir 500 images for cats/dogs
fnames=['{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    cat_img=os.path.join(cats_src,fname)
    dog_img=os.path.join(dogs_src,fname)
    cat_cpy=os.path.join(validation_cats_dir,fname)
    dog_cpy=os.path.join(validation_dogs_dir,fname)
    shutil.copyfile(cat_img,cat_cpy)
    shutil.copyfile(dog_img,dog_cpy)

#test_dir 500 images for cats/dogs
fnames=['{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    cat_img=os.path.join(cats_src,fname)
    dog_img=os.path.join(dogs_src,fname)
    cat_cpy=os.path.join(test_cats_dir,fname)
    dog_cpy=os.path.join(test_dogs_dir,fname)
    shutil.copyfile(cat_img,cat_cpy)
    shutil.copyfile(dog_img,dog_cpy)




