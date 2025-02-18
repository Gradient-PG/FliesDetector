import os
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod

import torch
import numpy as np
from ultralytics import YOLO

from tools.data_preperation import DataPreparation, BasicDataPreparation
from tools.image_cutter import ImageCutter
from tools.classifier import Classifier, ResnetClassifier
from tools.aligner import Aligner

class Pipeline(ABC):
    '''
    Abstract class providing basic structure for other pipeline implementations\n
    Neccesary conditions:
        - all model weights should be in models directiory
        - yolo weights should be named yolo_weights.pt
        - classifier weights should be named classifier_weights.pt
    '''
    def __init__(self, 
                 yolo_threshold: Optional[float] = 0.25, 
                 classifier_threshold: Optional[float] = 0.5,
                 resize_first: Optional[Tuple[int, int]] = (768, 768)):
        '''
        :param yolo_threshold: float within <0;1> range that will determine the necessary confidence a model has to detect an object
        :param classifier_threshold: float within <0;1> range that will determine the necessary confidence a model has in 
        classification, if not met then the detected object will be descibed as Unsure
        :param resize_first: single integer or a tuple of two integers, identicates the size of the input image to be resized into
        '''
        self._PATH = os.path.join(os.getcwd(), 'project', 'code', 'yolo2class_pipeline')
        # If gpu is available use it for inference
        # cuda - default gpu
        # cuda:x - x-th gpu
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.config = {
            'models_path': os.path.join(self._PATH, 'models'),  # Path to models folder
            'images_path': os.path.join(self._PATH, 'images'),  # Path where all images should be saved
            'detection_threshold': yolo_threshold,
            'classifier_threshold': classifier_threshold,
            'image_size': resize_first
        }
        # create output directory, if not exists
        os.makedirs(self.config['images_path'], exist_ok=True)


    def __call__(self, path: List[str]) -> None:
        '''
        Method where the whole pipeline flow should be placed
        :param path: list of images to perform the pipeline on
        '''
        return super().__call__()


class Yolo2ClassifierPipeline(Pipeline):
    def __init__(self, yolo_threshold: Optional[float] = None, classifier_threshold: Optional[float] = None):
        super().__init__(yolo_threshold, classifier_threshold)
        # DataPreparation (Resize, Normalize, ToTensor)
        self.data_preperation: DataPreparation = BasicDataPreparation()
        # Pretrained yolo model
        self.yolo = YOLO(os.path.join(self.config['models_path'], 'yolo_weights.pt'), 
                         task='detect')
        self.yolo.to(self.device)
        self.yolo.eval()
        # Class to cut all detections out of image, and transform them appropriately
        self.image_cutter = ImageCutter()
        # Classifier to perform inference on cut images
        self.classifier: Classifier = ResnetClassifier(os.path.join(self.config['models_path'], 'classifier_weights.pt'), 
                                                       self.device,
                                                       self.config['classifier_threshold'])
        # Class to align bounding boxes and corresponding classifications on the original image
        self.aligner = Aligner(self.config['images_path'])
    
    def __call__(self, path: List[str]) -> None:
        # Perform preperation on given images
        images = self.data_preperation(path, resize_size=self.config['image_size'])
        # Get bounding boxes detected by yolo from images
        # visualize=True -> lets you see how model is "seeing" images
        detections = self.yolo(images, 
                               conf=self.config['detection_threshold'], 
                               imgsz=self.config['image_size'])
        # Cut all bounding boxes from images
        cutter_output = self.image_cutter(detections, self.classifier.get_transformations())
        # Classify each cut image
        classifier_output = self.classifier(cutter_output)
        # Join everything into one picture (image + bbox + label) + counter of labels
        self.aligner(images, detections, classifier_output)

if __name__ == '__main__':
    pipeline = Yolo2ClassifierPipeline(yolo_threshold=0.28, classifier_threshold=0.5)
    example_images = [os.path.join(os.getcwd(), 'project', 'data', 'label_studio', 'male-normal.female-normal', 'images', '3f8d5aaf-IMG_5278.jpg'),
                      os.path.join(os.getcwd(), 'project', 'data', 'label_studio', 'male-normal.female-normal', 'images', '7d410af7-IMG_7023.jpg')]
    pipeline(example_images)