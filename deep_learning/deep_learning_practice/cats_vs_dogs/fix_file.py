import os
from PIL import Image

def is_valid(path):
    valid=True
    try:
        Image.open(path).load()
    except OSError:
        valid=False
    return valid

fpath='./Downloads/PetImages/Dog'
file_names=os.listdir('./Downloads/PetImages/Dog')
for fname in file_names:
    if not is_valid(os.path.join(fpath,fname)):
        print(fname)
