import numpy as np
from keras import losses
from keras import backend as K

def smoothL1(ypred, ytrue):
	x = K.abs(ytrue - y_pred)
	x = tf.where(tf.less(x, 1.0), 0.5 * x ** 2, x - 0.5)
	return K.sum(x, axis=-1)

def cls_loss(ypred, ytrue):
	return losses.categorical_crossentropy(y_true,y_pred)