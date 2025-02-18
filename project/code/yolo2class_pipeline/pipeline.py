import os
import argparse
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
                 resize_first: Optional[Tuple[int, int]] = (768, 768),
                 save_cut: Optional[bool] = False):
        '''
        :param yolo_threshold: float within <0;1> range that will determine the necessary confidence a model has to detect an object
        :param classifier_threshold: float within <0;1> range that will determine the necessary confidence a model has in 
        classification, if not met then the detected object will be descibed as Unsure
        :param resize_first: single integer or a tuple of two integers, identicates the size of the input image to be resized into
        :param save_cut: True if you want to save cut bounding boxes from images
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
            'image_size': resize_first,
            'save_cut': save_cut
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
    def __init__(self, 
                 yolo_threshold: Optional[float] = 0.25, 
                 classifier_threshold: Optional[float] = 0.5,
                 resize_first: Optional[Tuple[int, int]] = (768, 768),
                 save_cut: Optional[bool] = False):
        super().__init__(yolo_threshold, classifier_threshold, resize_first, save_cut)
        # DataPreparation (Resize, Normalize, ToTensor)
        self.data_preperation: DataPreparation = BasicDataPreparation()
        # Pretrained yolo model
        self.yolo = YOLO(os.path.join(self.config['models_path'], 'yolo_weights.pt'), 
                         task='detect')
        self.yolo.to(self.device)
        self.yolo.eval()
        # Class to cut all detections out of image, and transform them appropriately (optionally save)
        self.image_cutter = ImageCutter(self.config['images_path']) if self.config['save_cut'] else ImageCutter()
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


def main(input_images: List[str], input_yolo_threshold: float, input_classifier_threshold: float, input_save_cut: bool) -> None:
    pipeline = Yolo2ClassifierPipeline(yolo_threshold=input_yolo_threshold, 
                                       classifier_threshold=input_classifier_threshold,
                                       save_cut=input_save_cut)
    pipeline(input_images)

# CLI
# For further information execute
# pipeline.py -h or pipeline.py --help
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Yolo2Classifier pipeline on images")
    
    parser.add_argument("image_paths", type=str, nargs="+", help="List of image paths")
    parser.add_argument("--yolo_threshold", type=float, default=0.25, help="Confidence threshold for yolo detector")
    parser.add_argument("--classifier_threshold", type=float, default=0.5, help="Confidence threshold for classifier")
    parser.add_argument("--save_cut", type=bool, default=False, help="True if cut bounding boxes should be saved")

    args = parser.parse_args()

    main(args.image_paths, args.yolo_threshold, args.classifier_threshold, args.save_cut)