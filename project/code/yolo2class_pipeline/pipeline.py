import os
import argparse
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod

import torch
import numpy as np
from ultralytics import YOLO

from tools.data_preperation import DataPreparation, BasicDataPreparation
from tools.image_cutter import ImageCutter
from tools.classifier import Classifier, VitClassifier
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
                 classifier_threshold: Optional[float] = 0.35,
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
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.config = {
            'models_path': os.path.join(self._PATH, 'models'),  # Path to models folder
            'images_path': os.path.join(self._PATH, 'images'),  # Path where all images should be saved
            'detection_threshold': yolo_threshold,
            'classifier_threshold': classifier_threshold,
            'image_size': resize_first,
            'save_cut': save_cut
        }
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
                 classifier_threshold: Optional[float] = 0.35,
                 resize_first: Optional[Tuple[int, int]] = (768, 768),
                 save_cut: Optional[bool] = False):
        super().__init__(yolo_threshold, classifier_threshold, resize_first, save_cut)

        self.data_preperation: DataPreparation = BasicDataPreparation()

        self.yolo = YOLO(os.path.join(self.config['models_path'], 'detector.pt'), task='detect')
        self.yolo.to(self.device)
        self.yolo.eval()
        
        self.image_cutter = ImageCutter(self.config['images_path']) if self.config['save_cut'] else ImageCutter()

        self.classifier: Classifier = VitClassifier(self.config['models_path'], 
                                                       self.device,
                                                       self.config['classifier_threshold'])
        
        self.aligner = Aligner(self.config['images_path'])
    
    def __call__(self, path: List[str]) -> None:
        images = self.data_preperation(path, resize_size=self.config['image_size'])

        detections = self.yolo(images, 
                               conf=self.config['detection_threshold'], 
                               imgsz=self.config['image_size'])

        cutter_output = self.image_cutter(detections, self.classifier.get_transformations())
        classifier_output = self.classifier(cutter_output)
        
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
    parser.add_argument("--classifier_threshold", type=float, default=0.35, help="Confidence threshold for classifier")
    parser.add_argument("--save_cut", type=bool, default=False, help="True if cut bounding boxes should be saved")

    args = parser.parse_args()

    main(args.image_paths, args.yolo_threshold, args.classifier_threshold, args.save_cut)