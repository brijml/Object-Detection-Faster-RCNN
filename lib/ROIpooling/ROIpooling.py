import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
import xml.etree.ElementTree as ET
import scipy.io as spio
from keras import backend as K
# from .data_utils import parse
from keras.layers import MaxPooling2D

def parse(xml_file):
	labels = []
	tree = ET.parse(xml_file)
	root = tree.getroot()
	for child in root:
		if child.tag == 'object':
			name = child[0].text
			x1,y1,x2,y2 = int(child[4][0].text),int(child[4][1].text),int(child[4][2].text),int(child[4][3].text)
			labels.append({name:[x1,y1,x2,y2]})

	return labels

def resize(image,labels,arr):
	'''
	resizing image so as to have largest dimension to be of length 1000 (maintaing aspect ratio)
	'''
	m,n = image.shape[:2]
	a = float(m)/n
	if m>n : m1,n1 = 1000,int(1000/a)
	else: m1,n1 = int(1000*a),1000
	rm,rn = int(m1/m),int(n1/n)
	image = cv2.resize(image,(n1,m1))
	
	# applying same resize to label co-ordinates, the co-ordinates are in convention (x,y) and not (row,column)
	for label in labels:
		k = label.values()[0] 
		label[label.keys()[0]] = [int(k[0]*rn),int(k[1]*rm),int(k[2]*rn),int(k[3]*rm)]

	# the co-ordinates are in (row,column)
	arr[:,0] = arr[:,0]*rm
	arr[:,1] = arr[:,1]*rn
	arr[:,2] = arr[:,2]*rm
	arr[:,3] = arr[:,3]*rn
	return image,labels,arr

def forward(img,proposals_image):
	input_img = K.variable(img)
	model = VGG16(weights='imagenet', include_top=False, pooling=None)
	#The point of this code is I dont know what is wrong with keras, ever if I say pooling is None it is pooling
	#Yaar, kya hua hai keras ko??
	new_model = Sequential()
	for layer in model.layers[:-1]:
		new_model.add(layer)
	conv_feature_map = new_model(input_img)
	one_proposal = proposals_image[500]
	r,c,h,w = one_proposal[0],one_proposal[1],one_proposal[2],one_proposal[3]
	r1,c1,h1,w1 = r%16,c%16,int(h/16),int(w/16)
	roi = ROIpool()
	roi.forward(conv_feature_map,r1,c1,h1,w1)
	return

class ROIpool(object):
	"""docstring for ROIpool"""
	def __init__(self):
		super(ROIpool, self).__init__()
		self.H = 7
		self.W = 7

	def forward(self, conv_feature_map, r, c, h, w):
		window = conv_feature_map[:,r:r+h,c:c+w,:]
		print window.shape
		pool_layer = MaxPooling2D(pool_size=(h/self.H+1,w/self.W))
		pooled = pool_layer(window)
		print pooled.shape

		

if __name__ == '__main__':
	#To complete forward and backward pass for a single image
	name = '000021'
	proposals = spio.loadmat('/home/brij/selective_search_data/voc_2007_trainval.mat')
	image_name = '/home/brij/Desktop/github_projects/obj_det/datasets/VOCdevkit/VOC2007/JPEGImages/'+name+'.jpg'
	annotation_file = '/home/brij/Desktop/github_projects/obj_det/datasets/VOCdevkit/VOC2007/Annotations/'+name+'.xml'
	img = cv2.imread(image_name)
	labels = parse(annotation_file)
	index = np.where(proposals['images'] == name)[0][0]
	proposals_image = proposals['boxes'][:,index][0]
	img,labels,proposals_image = resize(img,labels,proposals_image)
	# for val in labels:
	# 	v = val.values()[0]
	# 	cv2.rectangle(img,(v[0],v[1]),(v[2],v[3]),(0,255,0),2)

	# plt.imshow(img)
	# plt.show()
	m,n,p = img.shape
	img = img.reshape(1,m,n,p)
	# print proposals_image[500]
	forward(img,proposals_image)
