import argparse
import os
import shutil
import json
import pandas as pd
import numpy as np
from PIL import Image
from typing import List, Tuple
from scripts.data_splitter import DataSplitter

import splitfolders

class ClassifierDatasetSplitter(DataSplitter):
    '''
    Splits whole dataset into train, test, val sets equally for each group
    '''
    def __init__(self, data_directory: str, ratio: Tuple[float] = (0.8, 0.1, 0.1), seed: int = 42):
        '''
        :param data_directory: path to the root folder
        :param ratio: Tuple of sizes each split should be (train, val, test)
        :param seed: Seed used for random generation of data splits
        '''

        self.ratio = ratio
        self.seed = seed
        self.PATHS = {
            'orig_data_path': os.path.join(data_directory, 'label_studio'),
            'cut_image_path': os.path.join(data_directory, 'classification', 'temp'),
            'final_dataset_path': os.path.join(data_directory, 'classification'),
            'classes_path': os.path.join(data_directory, 'info', 'classes.json')
        }
        self.classes = {}
    
    def prepare_folder(self) -> None:
        # prepare folder structure to satisfy splitter
        # dataset
        #   |
        #   |---class1
        #           - .jpg
        #           - .jpg
        #   |
        #   |---class2
        #           - .jpg
        #           - .jpg
        # Clear previous splits if they happened
        for folder in os.listdir(self.PATHS['final_dataset_path']):
            folder_path = os.path.join(self.PATHS['final_dataset_path'], folder)
            if (os.path.isdir(folder_path)):
                shutil.rmtree(folder_path)
        # Reads file with labels and corresponding class names
        with open(self.PATHS['classes_path']) as file:
            classes = json.load(file)
            self.classes = classes
            # Create folder for each class
            for class_name in classes.values():
                os.makedirs(os.path.join(self.PATHS['cut_image_path'], class_name), exist_ok=True)

    def process_image(self, image: str, annotation: str, path: str) -> None:
        '''
        Cut all bounding boxes from an image and save them to appropriate folders
        :param image: name of an image
        :param annotation: path to file with annotations corresponding to the given image, in YOLO format (e.g. class_id x_norm y_norm w_norm h_norm)
        :param path: path to the folder where image is located
        '''
        image_ = Image.open(os.path.join(path, 'images', image))

        image_arr = np.array(image_)
        height, width, _ = image_arr.shape
        with open(os.path.join(path, 'labels', annotation), mode='r') as ann:
            # Go through all labels cut out selected object and save it to proper folder
            for idx, detection in enumerate(ann.readlines()):
                detec = detection.split(" ")
                o_class = detec[0]
                x, y, w, h = map(float, detec[1:])
                y_min = int((y - h / 2) * height)
                y_max = int((y + h / 2) * height)
                x_min = int((x - w / 2) * width)
                x_max = int((x + w / 2) * width)
                cut_image = image_arr[y_min:y_max, x_min:x_max]
                cut_image = Image.fromarray(cut_image)
                cut_image.save(os.path.join(self.PATHS['cut_image_path'], self.classes[o_class], f"{image.split(".")[0]}_{idx}.jpg"))
    
    def create_csv(self, dataset_path: str) -> None:
        '''
        Generate .csv files for each dataset split (train,test,val)
        '''
        # test/train/val
        for folder in os.listdir(dataset_path):
            # Create .csv for this split
            annotation_csv = pd.DataFrame({'image_id': [], 'class': []})
            # female-normal/male-normal/male-white
            path_to_folder = os.path.join(dataset_path, folder)
            if (not os.path.isdir(path_to_folder)): continue
            for subfolder in os.listdir(path_to_folder):
                # ignore all files (we want only images folder)
                if (not os.path.isdir(os.path.join(path_to_folder, subfolder))): continue
                # @@@@.jpg/@@@@@.jpg
                for image in os.listdir(os.path.join(path_to_folder, subfolder)):
                    # @@@@.jpg, 0/1/2
                    annotation_csv.loc[len(annotation_csv.index)] = [image, list(self.classes.values()).index(subfolder)]
            # Save .csv
            annotation_csv.to_csv(os.path.join(path_to_folder, "labels.csv"), index=False)


    def __call__(self):
        # Create desired directory structure
        self.prepare_folder()
        # Go through all subfolders in folder with data, and cut every image
        for folder in os.listdir(self.PATHS['orig_data_path']):
            path = os.path.join(self.PATHS['orig_data_path'], folder)
            # For now ignore male-white folder, because of their wrong format
            if (folder == "male-white"): continue
            # Iterate through all images, annotations in folder
            for image_path, ann_path in zip(sorted(os.listdir(os.path.join(path, 'images'))), 
                                            sorted(os.listdir(os.path.join(path, 'labels')))):
                # Cut bboxes and save them
                self.process_image(image_path, ann_path, path)
        # Split prepared dataset
        splitfolders.ratio(self.PATHS['cut_image_path'], output=self.PATHS['final_dataset_path'], seed=self.seed, ratio=self.ratio, group_prefix=None, move=True)
        # Create .csv files for each split
        self.create_csv(self.PATHS['final_dataset_path'])
        # Delete redundant directory
        shutil.rmtree(self.PATHS['cut_image_path'])


def main(folder_path: str, ratio: List[float], seed: int) -> None:
    # dataset_path = os.path.join(os.getcwd(), 'project', 'data', 'label_studio')
    # DatasetSplitter(dataset_path)()
    ClassifierDatasetSplitter(folder_path, tuple(ratio), seed)()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
                                     Dataset Splitter for Classification purposes, splits data into\n
                                     parts specified in ratio argument
                                    """)
    
    parser.add_argument("--data_dir", type=str, default=f"{os.getcwd()}/project/data", help="Directory of the root folder with data")
    parser.add_argument("--ratio", type=float, nargs="+", default=[0.8, 0.1, 0.1] , help="How dataset should be split training_ratio val_ratio test_ratio")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for randomly splitting data")

    args = parser.parse_args()

    main(args.data_dir, args.ratio, args.seed)
    