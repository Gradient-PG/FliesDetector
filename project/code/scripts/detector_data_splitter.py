import os
import random
import shutil
import yaml
import argparse
from typing import List, Tuple, Optional
from scripts.data_splitter import DataSplitter

class DetectorDataSplitter(DataSplitter):
    '''
    Splits dataset into training, test, validation sets
    '''
    def __init__(self, data_directory: str, ratio: Tuple[float], seed: int = 42):
        '''
        :param source_path: path pointing to input data directory
        :param destination_path: path for splitted data
        :param ratio: Tuple of sizes each split should be (train, val, test)
        :param seed: Seed used for random generation of data splits
        '''
        self.source_dir = os.path.join(data_directory, 'label_studio')
        self.dest_dir = os.path.join(data_directory, 'detection')
        self.ratio = ratio
        self.seed = seed

        self.DEST_PATHS = {
            'dest_img_train': os.path.join(self.dest_dir, 'images', 'train'),
            'dest_img_val': os.path.join(self.dest_dir, 'images', 'val'),
            'dest_img_test': os.path.join(self.dest_dir, 'images', 'test'),
            'dest_lbl_train': os.path.join(self.dest_dir, 'labels', 'train'),
            'dest_lbl_val': os.path.join(self.dest_dir, 'labels', 'val'),
            'dest_lbl_test': os.path.join(self.dest_dir, 'labels', 'test'),
        }

        # Lists for collecting tuples like this: (img_path, label_path)
        self.file_pairs_list = []

        self.train_files = []
        self.test_files = []
        self.val_files = []

        # .YAML content
        self.yaml_content = {}

    def create_dirs(self):
        # Create destination directories structure
        for dir in self.DEST_PATHS.values():
            os.makedirs(dir, exist_ok=True)

    def collect_file_pairs(self):
        # Iterate through every subdirectory in source_dir
        for dir_name in os.listdir(self.source_dir):
            # Create path to certain directory contating images inside source_dir
            images_path = os.path.join(self.source_dir, dir_name, 'images')

            if os.path.isdir(images_path):
                # Iterate through images
                for image_name in os.listdir(images_path):
                    # We want only .jpg files, because they do not have EXIF metadata like .jpeg
                    if image_name.lower().endswith(('.jpg')):
                        # Create image path
                        image_path = os.path.join(images_path, image_name)

                        # Create corresponding path to image_name.txt which is label
                        # FROM YOLO DOCS: If there are no objects in an image, no *.txt file is required.
                        # So if the file does not exist do not append to pair and print warning
                        label_name = os.path.splitext(image_name)[0] + '.txt'
                        label_path = os.path.join(self.source_dir, dir_name, 'labels', label_name)

                        if os.path.exists(label_path):
                            self.file_pairs_list.append((image_path, label_path))
                        else:
                            print(f"Warning: No label found for {image_path}")

    
    def split_files(self, train_part: float, test_part: float, val_part: float, seed: int) -> None:
        '''
        Split all data into parts
        :param train_part: (0,1) float number representing portion of data to be in training split
        :param test_part: (0,1) float number representing portion of data to be in testing split
        :param val_part: (0,1) float number representing portion of data to be in validation split
        :param seed: int number to keep randomness 
        '''

        if train_part + test_part + val_part != 1.0:
            train_part, test_part, val_part = 0.8, 0.1, 0.1
            print(f"Warning: Splitting ratios don't add up to 1, dafaults used (0.8, 0.1, 0.1)")

        random.seed(seed)
        # Randomly distribute file pairs in the list so that every split is different
        random.shuffle(self.file_pairs_list)

        # Get how many images are in a dataset
        total_count = len(self.file_pairs_list)

        # Calculate indexes, that will split dataset
        train_count = int(total_count * train_part)
        test_count = int(total_count * test_part)
        val_count = int(total_count * val_part)

        # Create lists containing files for specified sets
        self.train_files = self.file_pairs_list[:train_count]
        self.test_files = self.file_pairs_list[train_count:train_count+test_count]
        self.val_files = self.file_pairs_list[train_count+test_count:train_count+test_count+val_count]

    def change_class_label(self, label_path: str) -> None:
        '''
        This function opens label file and changes all class indexes to 0 (detector is only meant to detect objects, ignoring their classes)
        :param label_path: path to the label file, where the replacing should happen
        '''
        # Open file
        with open(label_path, mode='r+') as file:
            # Take all lines from file
            lines = file.readlines()
            # Clear file
            file.truncate(0)
            # Set writing to the beginning
            file.seek(0)
            # replace only the first number (class_id) in every line with 0
            lines_changed = [line.replace(line[0], '0', 1) for line in lines]
            # write contents to file
            file.writelines(lines_changed)

    def copy_files(self, file_pairs: Tuple[str, str], dest_img: str, dest_lbl: str) -> None:
        '''
        Copy specified files (train, val or test) to their destination directory
        :param file_pairs: (image_path, label_path)
        :param dest_img: destination path for image to be copied to
        :param dest_lbl: destination path for label to be copied to
        '''
        for img, lbl in file_pairs:
            shutil.copy(img, dest_img)
            shutil.copy(lbl, dest_lbl)
            self.change_class_label(os.path.join(dest_lbl, lbl.split('/')[-1]))

    def create_yolo_yaml(self, filename: Optional[str] = 'detection_set.yaml') -> None:
        '''
        Create .yaml file for yolo training
        :param filename: desired name of the .yaml file
        '''
        classes = ['fly']

        self.yaml_content = {
            'path':'.',
            'train':'images/train',
            'val':'images/val',
            'test':'images/test',
            'nc': len(classes),
            'names': classes,
            'path': self.dest_dir
        }

        # Create .yaml file in specified directory
        yaml_path = os.path.join(self.dest_dir, filename)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.yaml_content, f)

        print(f"YAML config written to: {yaml_path}")

    def __call__(self) -> None:
        '''
        This function is running whole pipeline for splitting data accordingly to rules
        '''

        self.create_dirs()
        self.collect_file_pairs()
        self.split_files(*self.ratio, self.seed)

        self.copy_files(self.train_files, self.DEST_PATHS['dest_img_train'], self.DEST_PATHS['dest_lbl_train'])
        self.copy_files(self.val_files, self.DEST_PATHS['dest_img_val'], self.DEST_PATHS['dest_lbl_val'])
        self.copy_files(self.test_files, self.DEST_PATHS['dest_img_test'], self.DEST_PATHS['dest_lbl_test'])

        self.create_yolo_yaml()

        print("Splitting completed.")

def main(folder_path: str, ratio: List[float], seed: int) -> None:
    DetectorDataSplitter(folder_path, tuple(ratio), seed)()
    

# For more info
# python DetectorDataSplitter.py -h / --help
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
                                     Dataset Splitter for Detection/YOLO purposes, splits data into\n
                                     parts specified in ratio argument
                                    """)
    
    parser.add_argument("--data_dir", type=str, default=f"{os.getcwd()}/project/data", help="Directory of the root folder with data")
    parser.add_argument("--ratio", type=float, nargs="+", default=[0.8, 0.1, 0.1] , help="How dataset should be split training_ratio test_ratio val_ratio")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for randomly splitting data")

    args = parser.parse_args()

    main(args.data_dir, args.ratio, args.seed)