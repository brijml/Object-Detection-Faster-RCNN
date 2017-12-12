import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET
from random import shuffle
# import matplotlib.pyplot as plt
# from train_fastrcnn import visualise_minibatch


def show(img):
    plt.imshow(img)
    plt.show()
    return


def parse(xml_file):
    classes, boxes = [], []
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for child in root:
        if child.tag == 'object':
            name = child[0].text
            x1, y1, x2, y2 = int(child[4][0].text), int(child[4][1].text), int(child[4][2].text), int(child[4][3].text)
            classes.append(name)
            boxes.append([x1, y1, x2, y2])

    return classes, boxes


def union(au, bu, area_intersection):
    area_a = float(au[2] - au[0]) * float(au[3] - au[1])
    area_b = float(bu[2] - bu[0]) * float(bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return float(w)*float(h)


def intersection_over_union_score(a, b):
    # a and b should be (x1,y1,x2,y2)

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)


def transform(image, boxes, arr):
    m, n = image.shape[:2]
    image = cv2.flip(cv2.transpose(image), 1)
    for i, box in enumerate(boxes):
        boxes[i] = [m - box[3], box[0], m - box[1], box[2]]

    for i, box in enumerate(arr):
        arr[i] = [m - box[3], box[0], m - box[1], box[2]]

    return image, boxes, arr


def resize(image, *args):
    '''
    resizing image so as to have largest dimension to be of length 1000 (maintaing aspect ratio)
    '''
    arr = args[0]
    final_rois = np.zeros_like(arr)
    m, n = image.shape[:2]

    if m > n:
        image, boxes, arr = transform(image, boxes, arr)
        m, n = image.shape[:2]

    m1, n1 = 710, 1000
    rm, rn = float(m1) / m, float(n1) / n
    resized_image = cv2.resize(image, (n1, m1))

    final_rois[:, 0] = arr[:, 0] * rm
    final_rois[:, 1] = arr[:, 1] * rn
    final_rois[:, 2] = arr[:, 2] * rm
    final_rois[:, 3] = arr[:, 3] * rn

    # applying same resize to label co-ordinates, the co-ordinates are in convention (x,y) and not (row,column)
    if len(args) == 1:
        return resized_image, final_rois
        
    if len(args) == 2:
        boxes = args[1]
        resized_boxes = []
        for i, box in enumerate(boxes):
            resized_boxes.append([int(box[0] * rn), int(box[1] * rm), int(box[2] * rn), int(box[3] * rm)])


        return resized_image, resized_boxes, final_rois

