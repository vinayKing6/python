from keras.preprocessing import image

def single_image(path,size):
    img=image.load_img(path,target_size=size)
    x=image.img_to_array(img)/255
    x=x.reshape((1,)+x.shape)
    return x
