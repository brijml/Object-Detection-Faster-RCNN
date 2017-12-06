import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET
from random import shuffle
import matplotlib.pyplot as plt
from train_fastrcnn import visualise_minibatch


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


def intersection_over_union_score(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    try:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    except Exception as e:
        iou = 0

    if iou < 0:
        iou = 0

    return iou


def transform(image, boxes, arr):
    m, n = image.shape[:2]
    image = cv2.flip(cv2.transpose(image), 1)
    for i, box in enumerate(boxes):
        boxes[i] = [m - box[3], box[0], m - box[1], box[2]]

    for i, box in enumerate(arr):
        arr[i] = [m - box[3], box[0], m - box[1], box[2]]

    return image, boxes, arr


def resize(image, boxes, arr):
    '''
    resizing image so as to have largest dimension to be of length 1000 (maintaing aspect ratio)
    '''
    final_rois = np.zeros_like(arr)
    m, n = image.shape[:2]

    if m > n:
        image, boxes, arr = transform(image, boxes, arr)
        m, n = image.shape[:2]

    m1, n1 = 710, 1000
    rm, rn = float(m1) / m, float(n1) / n
    resized_image = cv2.resize(image, (n1, m1))

    # applying same resize to label co-ordinates, the co-ordinates are in convention (x,y) and not (row,column)
    resized_boxes = []
    for i, box in enumerate(boxes):
        resized_boxes.append([int(box[0] * rn), int(box[1] * rm), int(box[2] * rn), int(box[3] * rm)])

    final_rois[:, 1] = arr[:, 0] * rm
    final_rois[:, 0] = arr[:, 1] * rn
    final_rois[:, 3] = arr[:, 2] * rm
    final_rois[:, 2] = arr[:, 3] * rn

    return resized_image, resized_boxes, final_rois
