# This class implements basic transformations to be performed on input data

from abc import ABC, abstractmethod
from typing import List, Optional, Union, Sequence

from torchvision.transforms import v2
import torch

import os

from PIL import Image


class DataPreparation(ABC):
    def __init__(self):
        self.config = {
            'resize': (300, 300)
        }

    @abstractmethod
    def perform_transformations(self, /, 
                                image_paths: List[str | os.PathLike], 
                                additional_transformations: Optional[List[v2.Transform]] = None,
                                resize_size: Optional[Union[int, Sequence[int]]] = None) -> List[torch.Tensor]:
    # TODO moze zmienic na kwargsy
    # @abstractmethod
    # def perform_transformations(self, **kwargs) -> List[torch.Tensor]:
        """
        Perform basic transformations on input images (Resize, Normalize, ToTensor)
        :param image_paths: List of image paths, as strings or os.PathLike objects
        :param additional_transformations: List of additional transformations to be applied
        :param resize_size: Size to which images should be resized
        :return: List of transformed images as tensors
        """
        pass

    def __call__(self, /, 
                 image_paths: List[str | os.PathLike], 
                 additional_transformations: Optional[List[v2.Transform]] = None, 
                 resize_size: Optional[Union[int, Sequence[int]]] = None) -> List[torch.Tensor]:
        return self.perform_transformations(image_paths=image_paths, additional_transformations=additional_transformations, resize_size=resize_size)


# Basic transformations : Resize, Normalize
class BasicDataPreparation(DataPreparation):
    def __init__(self):
        super().__init__()
        self.transformations = v2.Compose([
            # v2.ToImage(),
            # v2.ToDtype(torch.float32, scale=True),
            v2.ToTensor()
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # TODO podmienic na ndarray
    def perform_transformations(
            self, /, 
            image_paths: List[str | os.PathLike], 
            additional_transformations: Optional[List[v2.Transform]] = None, 
            resize_size: Optional[Union[int, Sequence[int]]] = None) -> List[torch.Tensor]:
        # Perform basic transformations
        output = []
        for image_path in image_paths:
            # Load image
            image = Image.open(image_path)
            # Resize image
            if resize_size:
                image = v2.Resize(resize_size)(image)
            else:
                image = v2.Resize(self.config['resize'])(image)
            # Apply basic transformations
            # image = self.transformations(image)
            # If additional transformations are provided, perform them
            if additional_transformations:
                for transformation in additional_transformations:
                    image = transformation(image)
            output.append(image)
        # Return transformed image
        return output