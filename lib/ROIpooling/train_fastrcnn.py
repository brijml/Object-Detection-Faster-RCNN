import numpy as np
from roi_helpers import *
import scipy.io as spio
# from random import shuffle

POOL_SIZE = 7
DS_FACTOR = 16


def visualise_minibatch(img, batch):
    gboxes, pboxes = batch[1], batch[2]
    # print len(gboxes), len(pboxes)
    # print(len(gboxes), len(pboxes))
    for box in gboxes:
        if box is not None:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for box in pboxes:
        if box is not None:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # show(img)
    return


def one_hot(classes):

    probability = np.zeros((len(classes), 21), dtype=np.uint8)
    for i, class_ in enumerate(classes):
        probability[i, class_dict[class_]] = 1

    return probability


def parameterise(gbox, pbox):
    box = np.zeros(4, dtype=np.float32)
    pw, ph = pbox[2] - pbox[0], pbox[3] - pbox[1]
    gw, gh = gbox[2] - gbox[0], gbox[3] - gbox[1]
    box[0] = (gbox[0] - pbox[0]) / pw
    box[1] = (gbox[1] - pbox[1]) / ph
    box[2] = np.log(float(gw) / pw)
    box[3] = np.log(float(gh) / ph)
    return box


def create_array(classes, gboxes, pboxes):
    cls_array = one_hot(classes)
    bboxes_array = np.zeros((len(gboxes), 80), dtype=np.float32)

    for i, gbox in enumerate(gboxes):
        if gbox is not None:
            parametrised_box = parameterise(gbox, pboxes[i])
            # print("para box shape", parametrised_box.shape)
            start_idx = 4 * (class_dict[classes[i]] - 1)
            # print('start id', start_idx)
            bboxes_array[i, start_idx:start_idx + 4] = parametrised_box

    return cls_array, bboxes_array


def sample_minibatch(class_labels, gt_boxes, proposals_image):
    classes, bboxes, pboxes = [], [], []

    pcount = 0
    while pcount < 16:
        idx = np.random.randint(len(proposals_image))
        proposal = proposals_image[idx]
        proposal_width, proposal_height = proposal[3] - proposal[1], proposal[2] - proposal[0]
        if proposal_width > DS_FACTOR * POOL_SIZE and proposal_height > DS_FACTOR * POOL_SIZE:
            for i, gt_box in enumerate(gt_boxes):
                iou = intersection_over_union_score(gt_box, proposal)
                if iou > 0.5:
                    classes.append(class_labels[i])
                    bboxes.append(gt_box)
                    pboxes.append(proposal)
                    pcount += 1

    ncount = 0
    while ncount < 48:
        idx = np.random.randint(len(proposals_image))
        proposal = proposals_image[idx]
        proposal_width, proposal_height = proposal[3] - proposal[1], proposal[2] - proposal[0]
        if proposal_width > DS_FACTOR * POOL_SIZE and proposal_height > DS_FACTOR * POOL_SIZE:
            for i, gt_box in enumerate(gt_boxes):
                iou = intersection_over_union_score(gt_box, proposal)
                if iou >= 0.1 and iou < 0.5:
                    classes.append('background')
                    bboxes.append(None)
                    pboxes.append(None)
                    ncount += 1

    rand_idx = list(range(len(pboxes)))
    shuffle(rand_idx)

    temp_classes, temp_bboxes, temp_pboxes = [], [], []
    for i in rand_idx:
        temp_bboxes.append(bboxes[i])
        temp_classes.append(classes[i])
        temp_pboxes.append(pboxes[i])

    return classes, bboxes, pboxes


def train_on_batch(batch_class, batch_boxes):
    pass


if __name__ == '__main__':

    all_proposals = spio.loadmat('/home/brij/selective_search_data/voc_2007_trainval.mat')
    class_dict = {"background": 0, "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4, "bottle": 5, "bus": 6, "car": 7,
                  "cat": 8, "chair": 9, "cow": 10, "diningtable": 11, "dog": 12, "horse": 13, "motorbike": 14,
                  "person": 15, "pottedplant": 16, "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}
    images_dir = '/home/brij/Desktop/github_projects/obj_det/datasets/VOCdevkit/VOC2007/JPEGImages/'
    annotations_dir = '/home/brij/Desktop/github_projects/obj_det/datasets/VOCdevkit/VOC2007/Annotations/'
    files = os.listdir(images_dir)
    N, R = 2, 64
    no_epochs = 10
    epoch = 0

    while epoch < no_epochs:
        shuffle(files)
        for i in range(0, len(files), N):
            # print('file no', i)
            files_batch = files[i:i + N]
            batch_class, batch_boxes, batch_pboxes = [], [], []
            imgs = []
            for file in files_batch:
                name = file.split('.')[0]
                img = cv2.imread(os.path.join(images_dir, file))
                xml_file = os.path.join(annotations_dir, name + '.xml')
                class_labels, gt_boxes = parse(xml_file)
                index = np.where(all_proposals['images'] == name)[0][0]
                proposals_image = all_proposals['boxes'][:, index][0]
                img, gt_boxes, proposals_image = resize(img, gt_boxes, proposals_image)
                # print('img shape', i, img.shape)

                batch = sample_minibatch(class_labels, gt_boxes, proposals_image)
                visualise_minibatch(img, batch)
                batch_class.extend(batch[0])
                batch_boxes.extend(batch[1])
                batch_pboxes.extend(batch[2])
                imgs.append(img)

            imgs_array = np.array(imgs)
            # print('batch classes', len(batch_class))
            batch_class_array, batch_boxes_array = create_array(batch_class, batch_boxes, batch_pboxes)
            # train_on_batch(imgs_array, batch_class_array, batch_boxes_array)
        epoch += 1
        # print('epoch', epoch)
