from typing import List
from collections import Counter
import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from ultralytics.engine.results import Results


class Aligner:
    def __init__(self, path: str | os.PathLike):
        self.config = {
            'path': path,
            'color': 'blue',
            'font_color': 'white'
        }

    def align(self, 
              original_images: List[Image.Image],
              bounding_boxes: List[Results],
              classifications: List) -> None: # TODO uzupelnic dobrym typowaniem
        """
        Aligns classification output with detection output on the input image
        :param original_images: List of original images
        :param bounding_boxes: List of bounding boxes
        :param classifications: List of classifications results
        :return: None, image is saved
        """
        for idx, data in enumerate(zip(original_images, bounding_boxes, classifications)):
            image_path, bbox, classify = data
            # Show image
            plt.figure(figsize=(10, 10))
            plt.imshow(image_path)

            counter = Counter()

            # Loop through detections
            for box, conf, class_id in zip(bbox.boxes.xyxy, bbox.boxes.conf, classify):
                # Draw bounding boxes with probabilities

                # Bounding boxes
                x_min, y_min, x_max, y_max = map(int, box)
                plt.gca().add_patch(
                    plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                edgecolor=self.config['color'], fill=False, linewidth=2)
                )

                # Probabilities
                plt.gca().add_patch(
                    plt.Rectangle((x_max - 30, y_max - 20), 30, 20, 
                                edgecolor=self.config['color'], facecolor=self.config['color'], fill=True, linewidth=2)
                )
                plt.text(x_max - 30, y_max - 5, round(conf.item(), 2), color=self.config['font_color'], fontsize=8)

                # Draw classification results
                plt.gca().add_patch(
                    plt.Rectangle((x_min, y_min), x_max - x_min, 15, 
                                edgecolor=self.config['color'], facecolor=self.config['color'], fill=True, linewidth=2)
                )
                plt.text(x_min, y_min + 10, f"{class_id[0]}: {round(class_id[1], 2)}", color=self.config['font_color'], fontsize=8)
                counter[class_id[0]] += 1
            
            # Draw counter for all classes
            width, _ = image_path.size
            padding_horizontal = 5
            padding_vertical = 15
            text_height = 20
            legend_height = text_height * len(counter)
            legend_width = len(max(counter.keys(), key=lambda x: len(x))) * 8 + 2 * padding_vertical
            plt.gca().add_patch(
                    plt.Rectangle((width - legend_width, 0), legend_width, legend_height, 
                                edgecolor=self.config['color'], facecolor=self.config['color'], fill=True, linewidth=2)
                )
            
            # Iterate through counter and write each line (name: amount)
            for i, (name, amount) in enumerate(counter.items()):
                plt.text(width - legend_width + padding_horizontal, padding_vertical + (i * text_height), f"{name}: {amount}", color=self.config['font_color'], fontsize=10)

            # Save plot to file
            plt.axis('off')
            plt.savefig(os.path.join(self.config["path"], f"test{idx}.png"))
    
    def __call__(self,
                original_images: List[Image.Image],
                bounding_boxes: List[Results],
                classifications: List):
        self.align(original_images, bounding_boxes, classifications)

