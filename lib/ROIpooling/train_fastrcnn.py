import numpy as np
from roi_helpers import *
import scipy.io as spio
from new_model import fast_rcnn
from keras.models import Model,load_model
from keras.layers import Input
from keras import optimizers
from loss import *
import argparse

POOL_SIZE = 7
DS_FACTOR = 16

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Fast RCNN training")
    parser.add_argument("--pretrained", type=int, help="To load an existing model or not")
    parser.add_argument("--modelfile", type=str, help="Path to the file with model weights.")
    return parser.parse_args()


def visualise_minibatch(img, batch):
    if len(batch) == 3:
        gboxes, pboxes = batch[1], batch[2]
    else:
        gboxes, pboxes = batch, None

    for box in gboxes:
        if box is not None:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if pboxes is not None:
        for box in pboxes:
            if box is not None:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # show(img)
    return


def one_hot(classes_list):

    probability = np.zeros((len(classes_list), len(classes_list[0]), 21), dtype=np.uint8)
    for i, classes in enumerate(classes_list):
        for j, class_ in enumerate(classes):
            probability[i, j, class_dict[class_]] = 1

    print probability[0][0][0]
    return probability


def parameterise(gbox, pbox):
    box = np.zeros(4, dtype=np.int16)
    pw, ph = pbox[2] - pbox[0], pbox[3] - pbox[1]
    gw, gh = gbox[2] - gbox[0], gbox[3] - gbox[1]
    box[0] = (gbox[0] - pbox[0]) / pw
    box[1] = (gbox[1] - pbox[1]) / ph
    box[2] = np.log(float(gw) / pw)
    box[3] = np.log(float(gh) / ph)
    return box


def create_array(classes, gboxes_list, pboxes_list):
    cls_array = one_hot(classes)
    bboxes_coords = np.zeros((len(gboxes_list),len(gboxes_list[0]), 80), dtype=np.int16)
    bboxes_labels = np.zeros((len(gboxes_list),len(gboxes_list[0]), 80), dtype=np.int16)
    pboxes_array = np.zeros((len(gboxes_list),len(gboxes_list[0]), 4), dtype=np.int16)

    for i,gboxes in enumerate(gboxes_list):
        for j, gbox in enumerate(gboxes):
            if gbox is not None:
                parametrised_box = parameterise(gbox, pboxes_list[i][j])
                pboxes_array[i,j,:] = pboxes_list[i][j]
                start_idx = 4 * (class_dict[classes[i][j]] - 1)
                bboxes_coords[i, j, start_idx:start_idx + 4] = parametrised_box
                bboxes_labels[i, j, start_idx:start_idx + 4] = [1,1,1,1]

    bboxes_array = np.concatenate([bboxes_labels, bboxes_coords], axis=1)
    return cls_array, bboxes_array, pboxes_array


def create_iou_array(gt_boxes, proposals_image):

    iou_array = np.zeros((len(proposals_image), len(gt_boxes)), np.float32)
    for i, proposal in enumerate(proposals_image):
        #proposal_width, proposal_height = proposal[3] - proposal[1], proposal[2] - proposal[0]
        #if proposal_width > DS_FACTOR * POOL_SIZE and proposal_height > DS_FACTOR * POOL_SIZE:
        for j, gt_box in enumerate(gt_boxes):
            iou_array[i, j] = intersection_over_union_score(gt_box, proposal)

    return iou_array


def sample_minibatch(class_labels, gt_boxes, proposals_image):
    classes, bboxes, pboxes = [], [], []

    iou_array = create_iou_array(gt_boxes, proposals_image)
    indices_positive = np.where(iou_array > 0.5)
    no_positive = len(indices_positive[0])

    while len(classes) < 16:
        rand_idx = np.random.randint(low=0, high=no_positive)
        p_idx, g_idx = indices_positive[0][rand_idx], indices_positive[1][rand_idx]
        classes.append(class_labels[g_idx])
        bboxes.append(gt_boxes[g_idx])
        pboxes.append(proposals_image[p_idx])

    indices_negative = np.where(np.logical_and(0.1 <= iou_array, iou_array < 0.5))
    no_negative = len(indices_negative[0])

    while len(classes) < 64 :
        rand_idx = np.random.randint(low=0, high=no_negative)
        p_idx, g_idx = indices_negative[0][rand_idx], indices_negative[1][rand_idx]
        classes.append('background')
        bboxes.append(None)
        pboxes.append(proposals_image[p_idx])

    rand_idx = list(range(len(pboxes)))
    shuffle(rand_idx)

    temp_classes, temp_bboxes, temp_pboxes = [], [], []
    for i in rand_idx:
        temp_bboxes.append(bboxes[i])
        temp_classes.append(classes[i])
        temp_pboxes.append(pboxes[i])

    return temp_classes, temp_bboxes, temp_pboxes


if __name__ == '__main__':

    args = get_arguments()
    all_proposals = spio.loadmat('/home/mplubuntu/Desktop/dump/Object-Detection-Faster-RCNN/selective_search_data/voc_2007_train.mat')
    class_dict = {"background": 0, "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4, "bottle": 5, "bus": 6, "car": 7,
                  "cat": 8, "chair": 9, "cow": 10, "diningtable": 11, "dog": 12, "horse": 13, "motorbike": 14,
                  "person": 15, "pottedplant": 16, "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}
    images_dir = '/home/mplubuntu/Downloads/VOC2007/JPEGImages/'
    annotations_dir = '/home/mplubuntu/Downloads/VOC2007/Annotations/'
    files = os.listdir(images_dir)
    N, R = 2, 64
    no_epochs = 10
    epoch = 0

    all_loss = []
    best_loss = np.Inf

    if args.pretrained == 1:
        model = load_model(args.modelfile)

    else:
        img_input = Input(shape=(None, None, 3))
        roi_input = Input(shape=(None, 4))
        model = Model([img_input, roi_input], fast_rcnn(img_input, roi_input))
        model.compile(loss=[cls_loss, smoothL1],optimizer=optimizers.RMSprop(lr=1e-5, decay=0.0001),  metrics=['accuracy'])

    count,batch_count = 0,0 
    while epoch < no_epochs:
        shuffle(files)
        batch_class, batch_boxes, batch_pboxes,imgs = [], [], [],[]
        for i,f in enumerate(files):
            # files_batch = files[i:i + N]

            try:
                # for file in files_batch:
                name = f.split('.')[0]
                img = cv2.imread(os.path.join(images_dir, f))
                xml_file = os.path.join(annotations_dir, name + '.xml')
                class_labels, gt_boxes = parse(xml_file)

                index = np.where(all_proposals['images'] == name)[0][0]

                proposals_image = all_proposals['boxes'][:, index][0]
                img, gt_boxes, proposals_image = resize(img, gt_boxes, proposals_image)
                
                
                batch = sample_minibatch(class_labels, gt_boxes, proposals_image)
                #visualise_minibatch(img, batch)
                batch_class.append(batch[0])
                batch_boxes.append(batch[1])
                batch_pboxes.append(batch[2])
                imgs.append(img)

                print len(imgs),N
                if len(imgs) == N:
                    print "one batch"
                    imgs_array = np.array(imgs)
                    print imgs_array.shape
                    batch_class_array, batch_boxes_array,batch_roi_array = create_array(batch_class, batch_boxes, batch_pboxes)
                    print batch_class_array.shape, batch_boxes_array.shape, batch_roi_array.shape
                    batch_count+=1
                    print batch_count
                    try:
                        batch_loss = model.train_on_batch(x=[imgs_array, batch_roi_array], y=[batch_class_array, batch_boxes_array])
                    except Exception as e:
                        print e
                    print batch_loss
                    all_loss.append(batch_loss)
                    batch_class, batch_boxes, batch_pboxes, imgs = [], [], [], []
                    print 'hi'
                else:
                    continue

            except ValueError as v:
                count+=1

            except IndexError as i:
                continue

        epoch += 1
        if sum(all_loss) < best_loss:
            print('Total loss decreased from {} to {}, saving weights'.format(best_loss,sum(all_loss)))
            model.save('parameters/my-model.h5')
            best_loss = sum(all_loss)