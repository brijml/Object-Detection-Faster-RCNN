import numpy as np
from keras import losses
from keras import backend as K

epsilon = 1e-4

def smoothL1(y_true, y_pred):
	x = y_true[:, :, 80:] - y_pred
	x_abs = K.abs(x)
	x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
	return K.sum(y_true[:, :, :80] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :80])


def cls_loss(y_true, y_pred):
	return K.mean(losses.categorical_crossentropy(y_true[:, :, :], y_pred[:, :, :]))