# from keras.models import Sequential
# from keras.layers import Flatten, Dense, Dropout
# from keras.applications.vgg16 import VGG16
# from keras.layers import TimeDistributed


# def modified_vgg():
#     model = VGG16(weights='imagenet', include_top=False, pooling=None)
#     new_model = Sequential()
#     for layer in model.layers[:-1]:
#         new_model.add(layer)

#     new_model.add(roi_pooling_layer)  # add ROI pooling layer

#     f1 = TimeDistributed(Flatten(input_shape=(None, None)))  # flatten layer, I don't know if I'm even supposed to mention the imput_dim
#     f2 = TimeDistributed(Dense(4096, activation='relu', name='fc1'))
#     f3 = TimeDistributed(Dropout(0.5))  # dropout for regularization
#     f4 = TimeDistributed(Dense(4096, activation='relu', name='fc2'))
#     f5 = TimeDistributed(Dropout(0.5))  # dropout for regularization
#     '''
#     not sure about the kernel_initializer????
#     '''
#     f6 = TimeDistributed(Dense(20, activation='softmax', kernel_initializer='zero'))
#     f7 = TimeDistributed(Dense(4 * 20, activation='linear', kernel_initializer='zero'))

#     return

# ----------------------------------------------------------------------------------------------------------------------------------


from keras.applications.vgg16 import VGG16
from keras.layers import TimeDistributed
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from roipoolinglayer import RoiPoolingConv

def fast_rcnn(imgs, rois):
    vgg_model = VGG16(weights='imagenet', include_top=False, pooling=None)
    conv_feature_map = vgg_model(imgs)
    roi_pooling_layer = RoiPoolingConv(7, 128) # what to do???
    roi_out = roipoolinglayer(conv_feature_map, rois)
    f1 = TimeDistributed(Flatten(input_shape=(None, None)))(roi_out)  # dont know what to write here
    f2 = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(f1)
    f3 = TimeDistributed(Dropout(0.5))(f2)
    f4 = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(f3)
    f5 = TimeDistributed(Dropout(0.5))(f4)
    output1 = TimeDistributed(Dense(20, activation='softmax', kernel_initializer='zero'))(f5)
    output2 = TimeDistributed(Dense(4 * 20, activation='linear', kernel_initializer='zero'))(f5)

    return [output1, output2]
