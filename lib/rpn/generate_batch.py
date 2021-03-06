import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

def bb_intersection_over_union(boxA, boxB):
	#Taken from pyimagesearch
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = (xB - xA + 1) * (yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou

class Conv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
		super(Conv2d, self).__init__()
		padding = int((kernel_size - 1) / 2) if same_padding else 0
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
		self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
		self.relu = nn.ReLU(inplace=True) if relu else None

	def forward(self, x):
		x = self.conv(x)
		if self.bn is not None:
			x = self.bn(x)
		if self.relu is not None:
			x = self.relu(x)
		return x

class VGG16(nn.Module):
	"""docstring for VGG16"""
	def __init__(self, bn=None):
		super(VGG16, self).__init__()
		self.conv1 = nn.Sequential(Conv2d(3, 64, 3, same_padding=True, bn=bn),
		                           Conv2d(64, 64, 3, same_padding=True, bn=bn))
		self.pool1 = nn.MaxPool2d(2, return_indices=True)
		self.conv2 = nn.Sequential(Conv2d(64, 128, 3, same_padding=True, bn=bn),
		                           Conv2d(128, 128, 3, same_padding=True, bn=bn))
		self.pool2 = nn.MaxPool2d(2, return_indices=True)
		# network.set_trainable(self.conv1, requires_grad=False)
		# network.set_trainable(self.conv2, requires_grad=False)

		self.conv3 = nn.Sequential(Conv2d(128, 256, 3, same_padding=True, bn=bn),
		                           Conv2d(256, 256, 3, same_padding=True, bn=bn),
		                           Conv2d(256, 256, 3, same_padding=True, bn=bn))
		self.pool3 = nn.MaxPool2d(2, return_indices=True)
		self.conv4 = nn.Sequential(Conv2d(256, 512, 3, same_padding=True, bn=bn),
		                           Conv2d(512, 512, 3, same_padding=True, bn=bn),
		                           Conv2d(512, 512, 3, same_padding=True, bn=bn))
		self.pool4 = nn.MaxPool2d(2, return_indices=True)
		self.conv5 = nn.Sequential(Conv2d(512, 512, 3, same_padding=True, bn=bn),
		                           Conv2d(512, 512, 3, same_padding=True, bn=bn),
		                           Conv2d(512, 512, 3, same_padding=True, bn=bn))
		
	def forward(self, input_image):
		x = self.conv1(input_image)
		x,idx1 = self.pool1(x)
		x = self.conv2(x)
		x,idx2 = self.pool2(x)
		x = self.conv3(x)
		x,idx3 = self.pool3(x)
		x = self.conv4(x)
		x,idx4 = self.pool4(x)
		x = self.conv5(x)
		return x,idx1,idx2,idx3,idx4

def find_pixels_in_image(idx1,idx2,idx3,idx4):
	indices_img = []
	rows,cols = [],[]
	final_map = idx4.data.numpy()[0][0]
	idx3_numpy = idx3.data.numpy()[0][0]
	idx2_numpy = idx2.data.numpy()[0][0]
	idx1_numpy = idx1.data.numpy()[0][0]
	m,n = final_map.shape
	for i in range(m):
		for j in range(n):
			t = final_map[i,j]
			idx3i,idx3j = int(t/(2*n)),(t%(2*n))-1
			idx2i,idx2j = int(idx3_numpy[idx3i,idx3j]/(4*n)),(idx3_numpy[idx3i,idx3j]%(4*n))-1
			idx1i,idx1j = int(idx2_numpy[idx2i,idx2j]/(8*n)),(idx2_numpy[idx2i,idx2j]%(8*n))-1
			imi,imj = int(idx1_numpy[idx1i,idx1j]/(16*n)),(idx1_numpy[idx1i,idx1j]%(16*n))-1
			rows.append(imi);cols.append(imj)
	return (rows,cols)

def visualise(anchors):
	for anchor in anchors:
		cv2.rectangle(image,(int(anchor[0]),int(anchor[1])),(int(anchor[2]),int(anchor[3])),(0,255,0),2)

	plt.imshow(image)
	plt.show()
	return

def get_labels(gt,anchor):
	labels = []
	if bb_intersection_over_union(gt,anchor) > 0.7
		labels.append(1)
	else
		labels.append(0)
	return labels 

def find_anchors(image_pixels):
	rows,columns = image_pixels
	for i in range(len(rows)):
		anchors = generate_anchors(rows[i],columns[i])
		for anchor in anchors:
			cls_labels = get_labels(ground_truth,anchor)

	return


def get_values():
	size = []
	ratios = [0.5,1,2]
	scales = [128,256,512]
	for ratio in ratios:
		for scale in scales:
			size.append([scale*ratio,scale/ratio])

	return np.array(size)

def generate_anchors(x_ctr,y_ctr):
	x_min = x_ctr-(0.5*sizes[:,0])
	x_max = x_ctr+(0.5*sizes[:,0])
	y_min = y_ctr-(0.5*sizes[:,1])
	y_max = y_ctr+(0.5*sizes[:,1])
	return np.vstack((x_min,y_min,x_max,y_max)).T

if __name__ == '__main__':	
	all_weights = np.load('/home/brij/Desktop/github_projects/obj_det/VGG_imagenet.npy').item()
	network = VGG16(bn=None)
	vgg_parameters = network.state_dict()
	for k, v in vgg_parameters.items():
		i, j = int(k[4]), int(k[6]) + 1
		ptype = 'weights' if k[-1] == 't' else 'biases'
		key = 'conv{}_{}'.format(i, j)
		param = torch.from_numpy(all_weights[key][ptype])

		if ptype == 'weights':
			param = param.permute(3, 2, 0, 1)

		v.copy_(param)

	image = cv2.imread('/home/brij/Desktop/github_projects/obj_det/datasets/VOCdevkit/VOC2007/JPEGImages/000007.jpg')
	m,n,p = image.shape
	img = image.reshape(1,m,n,p)
	out,idx1,idx2,idx3,idx4 = network.forward(Variable(torch.from_numpy(img).permute(0,3,1,2).float()))
	corresponding_pixels = find_pixels_in_image(idx1,idx2,idx3,idx4)
	# map_ = np.zeros(image.shape[:2], dtype=np.uint8)
	# map_[corresponding_pixels] = 1
	# plt.subplot(121)
	# plt.imshow(image)
	# plt.subplot(122)
	# plt.imshow(map_, cmap='Greys')
	# plt.show()
	sizes = get_values()
	anchors = find_anchors(corresponding_pixels)