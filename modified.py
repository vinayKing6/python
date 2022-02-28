try:
    import cv2
    print('opencv except' )
except ImportError:
    print('missing opencv' ) 

try:
    import keras
    print('keras except') 
except ImportError:
    print('missing keras' )

try:
    import scipy
    print('scipy except' ) 
except ImportError:
    print('missing scipy' ) 

try:
    import matplotlib
    print('matplotlib except' )
except ImportError:
    print('missing matplotlib' )

try:
    import numpy
    print('numpy except' ) 
    print(numpy.__config__.show()) 
except ImportError:
    print('missing numpy' )

try:
    import tensorflow
    print('tensorflow except '+tensorflow.__version__ ) 
    if tensorflow.test.is_built_with_cuda:
        print('cuda supported' ) 
    else:
        print('missing cuda' ) 
except ImportError:
    print('missing tensorflow' )
