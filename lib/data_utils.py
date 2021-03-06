import cv2
import numpy as np
import xml.etree.ElementTree as ET
import json
import matplotlib.pyplot as plt
import os

def search_image(file):
	name = file.split('.')[0]
	for img_file in img_files:
		if img_file.split('.')[0] == name:
			return cv2.imread(os.path.join(img_directory,img_file)) 

def parse(xml_file):
	labels = {}
	tree = ET.parse(os.path.join(xml_directory,file))
	root = tree.getroot()
	for child in root:
		if child.tag == 'object':
			name = child[0].text
			x1,y1,x2,y2 = int(child[4][0].text),int(child[4][1].text),int(child[4][2].text),int(child[4][3].text)
			labels['name':[x1,y1,x2,y2]]

	return labels


if __name__ == '__main__':
	with open('config.json') as f:
		dict_ = json.load(f)

	img_directory = dict_['images_directory']
	xml_directory = dict_['xml_directory']
	img_files = os.listdir(img_directory)
	xml_files = os.listdir(xml_directory)
	read_xml()