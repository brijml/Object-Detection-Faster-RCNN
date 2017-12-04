import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET
from random import shuffle
import matplotlib.pyplot as plt


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
            # print('name', name)
            # print('1', child[4][0].text, child[4][1].text, child[4][2].text, child[4][3].text)
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


def resize(image, boxes, arr):
    '''
    resizing image so as to have largest dimension to be of length 1000 (maintaing aspect ratio)
    '''
    final_rois = np.zeros_like(arr)
    m, n = image.shape[:2]
    a = float(m) / n
    if m > n:
        # m1, n1 = 1000, int(1000 / a)
        m1, n1 = 1000, 710
    else:
        # m1, n1 = int(1000 * a), 1000
        m1, n1 = 710, 1000
    rm, rn = int(m1 / m), int(n1 / n)
    image = cv2.resize(image, (n1, m1))

    # applying same resize to label co-ordinates, the co-ordinates are in convention (x,y) and not (row,column)
    for i, box in enumerate(boxes):
        boxes[i] = [int(box[0] * rn), int(box[1] * rm), int(box[2] * rn), int(box[3] * rm)]

    # roi_height = arr[:,2]-arr[:,0]
    # roi_width = arr[:,3]-arr[:,1]
    # the co-ordinates are in (row,column)
    final_rois[:, 1] = arr[:, 0] * rm
    final_rois[:, 0] = arr[:, 1] * rn
    final_rois[:, 3] = arr[:, 2] * rm
    final_rois[:, 2] = arr[:, 3] * rn

    return image, boxes, final_rois
