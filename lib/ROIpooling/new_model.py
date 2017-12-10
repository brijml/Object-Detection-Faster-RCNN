from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import TimeDistributed
from keras.layers import Flatten, Dense, Dropout
from roipoolinglayer import RoiPoolingConv

vgg_model = VGG16(weights='imagenet', include_top=False, pooling=None)
new_model = Sequential()
for layer in vgg_model.layers[:-1]:
    new_model.add(layer)

def fast_rcnn(imgs, input_rois, nb_classes=21):
    conv_feature_map = new_model(imgs)
    out_roi_pool = RoiPoolingConv(7, 64)([conv_feature_map, input_rois])
    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
    out = TimeDistributed(Dropout(0.5))(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]
