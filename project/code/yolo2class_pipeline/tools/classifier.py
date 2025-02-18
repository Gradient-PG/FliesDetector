# TODO narazie robie zwykle pobieranie z import, potem wagi beda w pliku
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np

from torchvision.transforms import v2
import torch

import os
from abc import ABC, abstractmethod
from typing import List, Callable, Optional, Tuple

class Classifier(ABC):
    '''
    Class performing classification on given images 
    '''
    def __init__(self, 
                 model_path: str | os.PathLike,
                 device: torch.DeviceObjType,
                 confidence: Optional[float] = 0):
        '''
        :param model_path: name of the file with weights of the model, it should be present in models folder
        :param device: specifies on which device to perform operations
        :param confidence: float specifying confidence necessary to display result
        '''
        self.device = device
        self.config = {
            'model_path': model_path,
            'confidence': confidence
        }
    
    @abstractmethod
    def inference(self, images: List[np.ndarray[Image.Image]]) -> List[Tuple[str, float]]:
        '''
        :param images: list of PIL.Images, each one is a cut image 
        :return: List[]
        '''
        pass

    @abstractmethod
    def get_transformations(self) -> Callable:
        '''
        This method should return all necessary transformations for the classifier to work
        :return: function to transform images
        '''
        pass

    def __call__(self, images: np.ndarray[Image.Image]):
        return self.inference(images)


class ResnetClassifier(Classifier):
    def __init__(self, model_path: str | os.PathLike, device: torch.DeviceObjType, confidence: Optional[float] = None):
        super().__init__(model_path, device, confidence)
        # TODO For now classifier is not yet trained, in future exchange this
        self.weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=self.weights)
        # To this
        # self.model = torch.jit.load(model_path)
        # self.model.eval()
        self.model.to(self.device)
        self.model.eval()
        self.preprocess = self.weights.transforms()
    
    def inference(self, images: List[np.ndarray[Image.Image]]) -> List[Tuple[str, float]]:
        # result = [(class, conf), ...] 
        results = []
        for idx, full_image in enumerate(images):
            result_per_image = []
            # Check if there are any cut images taken from an image, if not then skip to next image
            if full_image.shape[0] == 0:
                continue
            input = torch.as_tensor(full_image)
            input = input.to(self.device)
            output = self.model(input).squeeze(0).softmax(0)
            # Check if output is a single tensor or a batch (necessary to handle single detection cases)
            if len(output.shape) == 1:
                output = output.unsqueeze(0)
            # Iterate through all results, infer class and confidence
            for result in output:
                class_id = result.argmax().item()
                score = result[class_id].item()
                category_name = self.weights.meta["categories"][class_id]
                if score > self.config['confidence']:
                    print(f"{idx}: {category_name}: {100 * score:.1f}%")
                    result_per_image.append((category_name, 100 * score))
                else:
                    print(f"{idx}: Unsure: {100 * score:.1f}%")
                    result_per_image.append(("Unsure", 100 * score))
            results.append(result_per_image)
        return results
    
    def get_transformations(self) -> Callable:
        return self.preprocess