from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import TimeDistributed
from keras.layers import Flatten, Dense, Dropout
from roipoolinglayer import RoiPoolingConv

vgg_model = VGG16(weights='imagenet', include_top=False, pooling=None)
new_model = Sequential()
for layer in vgg_model.layers[:-1]:
    new_model.add(layer)

def fast_rcnn(imgs, input_rois):
    conv_feature_map = new_model(imgs)
    out_roi_pool = RoiPoolingConv(7, 64)([conv_feature_map, input_rois])
    f1 = TimeDistributed(Flatten(input_shape=(None, None)))(out_roi_pool)
    f2 = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(f1)
    f3 = TimeDistributed(Dropout(0.5))(f2)
    f4 = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(f3)
    f5 = TimeDistributed(Dropout(0.5))(f4)
    output1 = TimeDistributed(Dense(20, activation='softmax', kernel_initializer='zero'))(f5)
    output2 = TimeDistributed(Dense(4 * 20, activation='linear', kernel_initializer='zero'))(f5)

    return [output1, output2]
