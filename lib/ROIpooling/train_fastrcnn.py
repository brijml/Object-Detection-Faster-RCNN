import numpy as np
from ROIpooling import *
from random import shuffle

def one_hot(classes):

	probability = np.zeros((len(classes),21), dtype=np.uint8)
	for i,class_ in enumerate(classes):
		probability[i,class_dict[class_]] = 1

	return probability

def parameterise(gbox, pbox):
	box = np.array(4, dtype = np.float32)
	pw,ph = pbox[2]-pbox[0], pbox[3]-pbox[1]
	gw,gh = gbox[2]-gbox[0], gbox[3]-gbox[1]
	box[0] = (gbox[0]-pbox[0])/pw
	box[1] = (gbox[1]-pbox[1])/ph
	box[2] = gw/pw
	box[3] = gh/ph
	return box

def create_array(classes, gboxes, pboxes):
	cls_array = one_hot(classes)
	bboxes_array = np.array((len(boxes), 80), dtype=np.float32)

	for i,gbox in enumerate(boxes):
		if gbox is not None:
			parametrised_box = parameterise(gbox, pboxes[i])
			start_idx = 4*class_dict[classes[i]]
			bboxes_array[i, start_idx:start_idx+4] = parametrised_box 

	return cls_array,bboxes_array

def sample_minibatch(labels, proposals_image):
	classes, bboxes, pboxes = [],[],[]
	class_labels = labels.keys()
	gt_boxes = labels.values()
	positives,negatives = 16,48
	
	while len(bboxes)<63:
		idx = np.random.randint(len(proposals_image))
		proposal = proposals_image[idx]
		if proposal_width>7 and proposal_height>7:
			for i,gt_box in enumerate(gt_boxes):
				iou = intersection_over_union_score(gt_box, proposal)
				if iou > 0.5:
					classes.append(class_labels[i])
					bboxes.append(gt_box)
					pboxes.append(proposal)
				elif iou>=0.1 and iou<0.5:
					classes.append('background')
					bboxes.append(None)
					pboxes.append(None)

	rand_idx = shuffle(range(64))
	temp_classes, temp_bboxes, temp_pboxes = [],[],[]
	for i in rand_idx:
		temp_bboxes.append(bboxes[i])
		temp_classes.append(classes[i])
		temp_pboxes.append(pboxes[i])

	return temp_classes,temp_bboxes

def train_on_batch(batch_class, batch_boxes):
	pass
	

if __name__ == '__main__':
	all_proposals = spio.loadmat('/home/brij/selective_search_data/voc_2007_trainval.mat')
	class_dict = {"background":0,"person":1,"bird":2, "cat":3, "cow":4, "dog":5, "horse":6,"sheep":7,
					"aeroplane":8, "bicycle":9, "boat":10, "bus":11, "car":12, "motorbike":13, "train":14,
					"bottle":15, "chair":16, "dining table":17, "potted plant":18, "sofa":19, "tv/monitor":20}
	images_dir = '/home/brij/Desktop/github_projects/obj_det/datasets/VOCdevkit/VOC2007/JPEGImages/'
	annotations_dir = '/home/brij/Desktop/github_projects/obj_det/datasets/VOCdevkit/VOC2007/Annotations/'
	files = os.listdir(images_dir)
	N,R = 2,64
	no_epochs = 10

	while epoch<no_epochs:
		shuffle(files)
		for i in range(0,len(files),N):
			files_batch = files[i:i+N]
			batch_class, batch_boxes, batch_pboxes = [],[],[]
			imgs = []
			for file in files_batch:
				name = file.split('.')[0]
				img = cv2.imread(os.path.join(images_dir,file))
				xml_file = os.path.join(annotations_dir,name+'.xml')
				gt_boxes = parse(xml_file)
				index = np.where(all_proposals['images'] == name)[0][0]
				proposals_image = proposals['boxes'][:, index][0]
				img, gt_boxes, proposals_image = resize(img, gt_boxes, proposals_image)
				batch = sample_minibatch(gt_boxes, proposals_image)
				batch_class.extend(batch[0]); batch_boxes.extend(batch[1]); batch_pboxes.extend(batch[2])
				imgs.append(img)

			imgs_array = np.array(imgs)
			batch_class_array, batch_boxes_array = create_array(batch_class, batch_boxes, batch_pboxes)
			train_on_batch(imgs_array, batch_class_array, batch_boxes_array)
		epoch+=1