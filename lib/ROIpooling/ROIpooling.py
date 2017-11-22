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
            x1, y1, x2, y2 = int(child[4][0].text), int(child[4][1].text), int(child[4][2].text), int(child[4][3].text)
            labels.append({name: [x1, y1, x2, y2]})

    return labels


def intersection_over_union_score(boxA, boxB):
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


def resize(image, labels, arr):
    '''
    resizing image so as to have largest dimension to be of length 1000 (maintaing aspect ratio)
    '''
    m, n = image.shape[:2]
    a = float(m) / n
    if m > n:
        m1, n1 = 1000, int(1000 / a)
    else:
        m1, n1 = int(1000 * a), 1000
    rm, rn = int(m1 / m), int(n1 / n)
    image = cv2.resize(image, (n1, m1))

    # applying same resize to label co-ordinates, the co-ordinates are in convention (x,y) and not (row,column)
    for label in labels:
        k = list(label.values())[0]
        label[list(label.keys())[0]] = [int(k[0] * rn), int(k[1] * rm), int(k[2] * rn), int(k[3] * rm)]

    # the co-ordinates are in (row,column)
    arr[:, 0] = arr[:, 0] * rm
    arr[:, 1] = arr[:, 1] * rn
    arr[:, 2] = arr[:, 2] * rm
    arr[:, 3] = arr[:, 3] * rn
    return image, labels, arr


def forward(img, proposals_image, labels):
    input_img = K.variable(img)
    model = VGG16(weights='imagenet', include_top=False, pooling=None)
    # The point of this code is I dont know what is wrong with keras, ever if I say pooling is None it is pooling
    # Yaar, kya hua hai keras ko??
    new_model = Sequential()
    for layer in model.layers[:-1]:
        new_model.add(layer)
    conv_feature_map = new_model(input_img)
    # print(img.shape)
    # print('test', np.min(proposals_image[:, 0]), np.min(proposals_image[:, 1]), np.max(proposals_image[:, 2]), np.max(proposals_image[:, 3]))

    iou_score_list = []
    temp_region_proposals_list = []
    for ind in range(proposals_image.shape[0]):
        one_proposal = proposals_image[ind]
        row_min, column_min, row_max, column_max = one_proposal[0] - 1, one_proposal[1] - 1, one_proposal[2] - 1, one_proposal[3] - 1
        r1, c1, r2, c2 = int(row_min / 16), int(column_min / 16), int(row_max / 16), int(column_max / 16)
        region_proposal_box = [column_min, row_min, column_max, row_max]
        for label in labels:
            '''
            For every region proposal determine iou with all labels
            '''
            GTbox = tuple(label.items())[0][1]
            iou_score = intersection_over_union_score(GTbox, region_proposal_box)
            if iou_score < 0.5:
                continue
            else:
                iou_score_list.append(iou_score)
                temp_region_proposals_list.append([r1, c1, r2, c2])
                break
                # print('found')
    # print('total no of proposals', proposals_image.shape[0])
    # print('no of proposals with more than 0.5 iou: ', len(temp_region_proposals_list))
    if len(temp_region_proposals_list) <= 64:
        selected_region_proposal_list = temp_region_proposals_list
    else:
        iou_score_list = np.array(iou_score_list)
        indices = iou_score_list.argsort()[-64:][::-1]
        # # #selected_region_proposal_list = temp_region_proposals_list[iou_score_list.argsort()[-64:][::-1]]  # selecting top 64 iou region proposals, numpy me bug he
        selected_region_proposal_list = [None] * 64

        for index, ind in enumerate(indices):
            selected_region_proposal_list[index] = temp_region_proposals_list[ind]

        for region in selected_region_proposal_list:
            '''
            Passing each of the selected region proposal for max pooling
            '''
            roi = ROIpool()
            roi.forward(conv_feature_map, region)
    return


class ROIpool(object):
    """docstring for ROIpool"""

    def __init__(self):
        super(ROIpool, self).__init__()
        self.H = 7
        self.W = 7

    def forward(self, conv_feature_map, region):
        [r1, c1, r2, c2] = region
        window = conv_feature_map[:, r1:r2, c1:c2, :]
        # print window.shape
        pool_layer = MaxPooling2D(pool_size=((r2 - r1) / self.H + 1, (c2 - c1) / self.W))
        pooled = pool_layer(window)
        # print(pooled.shape)


if __name__ == '__main__':
    # To complete forward and backward pass for a single image
    name = '000021'
    proposals = spio.loadmat('/home/brij/selective_search_data/voc_2007_trainval.mat')
    image_name = '/home/brij/Desktop/github_projects/obj_det/datasets/VOCdevkit/VOC2007/JPEGImages/' + name + '.jpg'
    annotation_file = '/home/brij/Desktop/github_projects/obj_det/datasets/VOCdevkit/VOC2007/Annotations/' + name + '.xml'
    img = cv2.imread(image_name)
    labels = parse(annotation_file)
    index = np.where(proposals['images'] == name)[0][0]
    proposals_image = proposals['boxes'][:, index][0]
    img, labels, proposals_image = resize(img, labels, proposals_image)
    # print(labels)
    # for val in labels:
    #   v = val.values()[0]
    #   cv2.rectangle(img,(v[0],v[1]),(v[2],v[3]),(0,255,0),2)

    # plt.imshow(img)
    # plt.show()
    m, n, p = img.shape
    img = img.reshape(1, m, n, p)
    # print proposals_image[500]
    forward(img, proposals_image, labels)
