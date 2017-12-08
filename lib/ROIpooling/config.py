class Config:

	def __init__(self, environ):

		if environ == 0:
			self.roi_mat_file = '/home/brij/selective_search_data/voc_2007_trainval.mat'
			self.images_path = '/home/brij/Desktop/github_projects/obj_det/datasets/VOCdevkit/VOC2007/JPEGImages/'
			self.annotations_path = '/home/brij/Desktop/github_projects/obj_det/datasets/VOCdevkit/VOC2007/Annotations/' 
			self.model_path = 'parameters/my-model.h5'

		else:
			self.roi_mat_file = '/home/brij/selective_search_data/voc_2007_trainval.mat'
			self.images_path = '/home/brij/datasets/VOCdevkit/VOC2007/JPEGImages/'
			self.annotations_path = '/home/brij/datasets/VOCdevkit/VOC2007/Annotations/' 
			self.model_path = 'parameters/my-model.h5'

