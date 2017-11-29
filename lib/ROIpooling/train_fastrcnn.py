import numpy as np
from ROIpooling import *
from random import shuffle
from minibatch_sampling import sample_minibatch

def sample_minibatch(labels, proposals_image):
	classes, bboxes = []
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
				elif iou>=0.1 and iou<0.5:
					classes.append('background')
					bboxes.append(None)

		else:
			continue

	rand_idx = shuffle(range(64))
	temp_classes, temp_bboxes = [],[]
	for i in rand_idx:
		temp_bboxes.append(bboxes[i])
		temp_classes.append(classes[i])

	return temp_classes,temp_bboxes

def train_on_batch(batch_class, batch_boxes):
	pass
	

if __name__ == '__main__':
	all_proposals = spio.loadmat('/home/brij/selective_search_data/voc_2007_trainval.mat')
	images_dir = '/home/brij/Desktop/github_projects/obj_det/datasets/VOCdevkit/VOC2007/JPEGImages/'
	annotations_dir = '/home/brij/Desktop/github_projects/obj_det/datasets/VOCdevkit/VOC2007/Annotations/'
	files = os.listdir(images_dir)
	N,R = 2,64
	no_epochs = 10

	while epoch<no_epochs:
		shuffle(files)
		for i in range(0,len(files),N):
			files_batch = files[i:i+N]
			batch_class, batch_boxes = [],[]
			for file in files_batch:
				name = file.split('.')[0]
				img = cv2.imread(os.path.join(images_dir,file))
				xml_file = os.path.join(annotations_dir,name+'.xml')
				gt_boxes = parse(xml_file)
				index = np.where(all_proposals['images'] == name)[0][0]
				proposals_image = proposals['boxes'][:, index][0]
				img, gt_boxes, proposals_image = resize(img, gt_boxes, proposals_image)
				batch = sample_minibatch(gt_boxes, proposals_image)
				batch_class.extend(batch[0]); batch_boxes.extend(batch[1])

			train_on_batch(batch_class, batch_boxes)
		epoch+=1