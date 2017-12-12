class Config:

    def __init__(self, environ):

        if environ == 0:
            self.train_roi_mat_file = '/home/brij/selective_search_data/voc_2007_trainval.mat'
            self.test_roi_mat_file = '/home/brij/selective_search_data/voc_2007_trainval.mat'
            self.images_path = '/home/brij/Desktop/github_projects/obj_det/datasets/VOCdevkit/VOC2007/JPEGImages/'
            self.annotations_path = '/home/brij/Desktop/github_projects/obj_det/datasets/VOCdevkit/VOC2007/Annotations/'
            self.model_path = 'parameters/my-model.h5'

        elif environ == 1:
            self.train_roi_mat_file = '/home/brij/selective_search_data/voc_2007_trainval.mat'
            self.test_roi_mat_file = '/home/brij/selective_search_data/voc_2007_trainval.mat'
            self.images_path = '/home/brij/datasets/VOCdevkit/VOC2007/JPEGImages/'
            self.annotations_path = '/home/brij/datasets/VOCdevkit/VOC2007/Annotations/'
            self.model_path = 'parameters/my-model.h5'

        elif environ == 2:
            self.train_roi_mat_file = '/Users/tejasbobhte/Desktop/tejas/FasterRCNN/Object-Detection-Faster-RCNN/lib/selective_search_data/voc_2007_trainval.mat'
            self.test_roi_mat_file = '/home/brij/selective_search_data/voc_2007_trainval.mat'
            self.images_path = '/Users/tejasbobhte/Downloads/VOCdevkit 2/VOC2007/JPEGImages/'
            self.annotations_path = '/Users/tejasbobhte/Downloads/VOCdevkit 2/VOC2007/Annotations/'
            self.model_path = 'parameters/my-model.h5'
