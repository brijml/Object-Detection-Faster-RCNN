import numpy as np
from keras.models import Model,load_model
import os,cv2
from roi_helpers import *

def get_arguments():
    parser = argparse.ArgumentParser(description="Fast RCNN testing")
    parser.add_argument("--modelfile", type=str, help="Path to the file with model weights.")
    parser.add_argument("--path", type=str, help="Path to the image or a directory")
    return parser:

def validate(p_boxes):
	valid_boxes = []
	invalid_boxes = []
	for box in p_boxes:
		area = box[0]*(box[2]-box[0])*box[0]*(box[2]-box[0])
		if area > 45000:
			valid_boxes.append(box)
		else:
			invalid_boxes.append(box)

	while len(valid_boxes)<64:
		rand_idx = np.random.randint(low=0, high=len(invalid_boxes))
		valid_boxes.append(invalid_boxes[rand_idx])

	return np.array(valid_boxes)

if __name__ == '__main__':
	args = get_arguments()
    C = Config(args.environ)
    all_proposals = spio.loadmat(C.test_roi_mat_file)
    class_dict = {"background": 0, "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4, "bottle": 5, "bus": 6, "car": 7,
                  "cat": 8, "chair": 9, "cow": 10, "diningtable": 11, "dog": 12, "horse": 13, "motorbike": 14,
                  "person": 15, "pottedplant": 16, "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}

    model = load_model(args.modelfile)

	files1 = os.listdir(args.path)
	files = [i for i in files1 if i.endswith('.jpg')]
	for i in range(0,len(files),2):
		files_batch = files[i:i+2]
		imgs = []
		for f in files_batch:
			name = f.split('.')[0]
			img = cv2.imread(os.path.join(args.path,f))
            index = np.where(all_proposals['images'] == name)[0][0]
            proposals_image = all_proposals['boxes'][:, index][0]
            img, p_boxes = resize(img, proposals_image)
            valid_boxes = validate(p_boxes)
            imgs.append(img)
            boxes.append(valid_boxes)

        imgs_array, p_array = np.array(boxes),np.array(boxes)
        m,n,p = p_array.shape
        
        all_predictions_cls,all_predictions_regr = np.zeros((2,n,21)), np.zeros((2,n,80))
        for i in range(0,n,64):
        	roi = p_array[:,i:i+64,:]
        	if roi.shape[1] < 64:
        		break
			
			predictions = model.predict_on_batch([imgs_array, p_array])
			all_predictions_cls[:,i:i+64,:] = predictions[0]
			all_predictions_regr[:,i:i+64,:] = predictions[1]
			
		non_maximum_suppression(imgs, all_predictions_cls, all_predictions_regr)
