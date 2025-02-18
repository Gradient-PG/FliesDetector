import os
from typing import List, Callable, Optional, Union
from math import floor, ceil

import numpy as np
import supervision as sv
from ultralytics.engine.results import Results
from torchvision.transforms import ToPILImage
from PIL import Image


class ImageCutter:
    '''
    Class implementing method to cut all provided bounding boxes from an 
    image and perform transformations neccessary for classification model
    '''
    def __init__(self, path: Optional[str] = None):
        '''
        :param path: Optional argument, provide a path where you want to save cut images
        '''
        self.config = {
            'path': path
        }

    def cut_images(self, predictions: List[Results], classifier_transformations: Callable) -> List[np.ndarray[Image.Image]]:
        '''
        Cut bounding boxes from list of images
        :param predictions: list of detections given by yolo inference
        :param classifier_transformations: set of transformations to apply on the cut images for further classification purpose
        :return: list[full batch] of np.ndarray[single image] of PIL.Image[cut bounding boxes from that image]
        '''
        cut_images = []
        # Iterate through all image predictions
        for img_idx, prediction in enumerate(predictions):
            prediction_images = []
            # Get original image (after transformations)
            orig_img = prediction.orig_img
            # Get coordinates of detected objects
            detections = sv.Detections.from_ultralytics(prediction)
            # Iterate through all detected objects
            for idx, detection in enumerate(detections):
                coords = detection[0]
                x1, y1, x2, y2 = coords
                # Cut detected object from original image
                detected_object = orig_img[floor(y1):ceil(y2), floor(x1):ceil(x2)]

                # Transform cut image into Image, and perform transformations for classifier                
                detec = Image.fromarray(detected_object)
                trans_detec = classifier_transformations(detec)

                # Save transformed image (optional)
                if self.config["path"] is not None:
                    x_detec = ToPILImage()(trans_detec)
                    x_detec.save(os.path.join(self.config["path"], f"detected_{img_idx}_{idx}.png"))
                
                prediction_images.append(trans_detec)
            prediction_images = np.array(prediction_images)
            cut_images.append(prediction_images)
        return cut_images


    def __call__(self, predictions: List[Results], classifier_transformations: Callable) -> List[np.ndarray[Image.Image]]:
        '''
        Uses cut_images function to perform cutting
        :param predictions: list of detections given by yolo inference
        :param classifier_transformations: set of transformations to apply on the cut images for further classification purpose
        :return: list[full batch] of np.ndarray[single image] of PIL.Image[cut bounding boxes from that image]
        '''
        return self.cut_images(predictions, classifier_transformations)