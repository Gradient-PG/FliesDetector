import os
import random
import shutil
import yaml
import argparse
from typing import List, Tuple, Optional

class DetectorDataSplitter:
    def __init__(self, source_path: str, destination_path: str):
        self.source_dir = source_path
        self.dest_dir = destination_path

        self.DEST_PATHS = {
            'dest_img_train': os.path.join(destination_path, 'images', 'train'),
            'dest_img_val': os.path.join(destination_path, 'images', 'val'),
            'dest_img_test': os.path.join(destination_path, 'images', 'test'),
            'dest_lbl_train': os.path.join(destination_path, 'labels', 'train'),
            'dest_lbl_val': os.path.join(destination_path, 'labels', 'val'),
            'dest_lbl_test': os.path.join(destination_path, 'labels', 'test'),
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

    def copy_files(self, file_pairs: Tuple[str, str], dest_img: str, dest_lbl: str) -> None:
        for img, lbl in file_pairs:
            shutil.copy(img, dest_img)
            shutil.copy(lbl, dest_lbl)

    def create_yolo_yaml(self, filename: Optional[str] = 'detection_set.yaml') -> None:
        classes = ['fly']

        self.yaml_content = {
            'path':'.',
            'train':'images/train',
            'val':'images/val',
            'test':'images/test',
            'nc': len(classes),
            'names': classes
        }

        # Create .yaml file in specified directory
        yaml_path = os.path.join(self.dest_dir, filename)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.yaml_content, f)

        print(f"YAML config written to: {yaml_path}")

    def run(self, 
            train_part: Optional[float] = 0.8, 
            test_part: Optional[float] = 0.1, 
            val_part: Optional[float] = 0.1,
            seed: Optional[int] = 42) -> None:
        self.create_dirs()
        self.collect_file_pairs()
        self.split_files(train_part, test_part, val_part, seed)

        self.copy_files(self.train_files, self.DEST_PATHS['dest_img_train'], self.DEST_PATHS['dest_lbl_train'])
        self.copy_files(self.val_files, self.DEST_PATHS['dest_img_val'], self.DEST_PATHS['dest_lbl_val'])
        self.copy_files(self.test_files, self.DEST_PATHS['dest_img_test'], self.DEST_PATHS['dest_lbl_test'])

        self.create_yolo_yaml()

        print("Splitting completed.")

def main(folder_path: str, ratio: List[float], seed: int) -> None:
    DetectorDataSplitter(folder_path, os.path.join(os.getcwd(), 'project', 'data', 'detection')).run(*ratio, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
                                     Dataset Splitter for Detection/YOLO purposes, splits data into\n
                                     parts specified in ratio argument
                                    """)
    
    parser.add_argument("--image_dir", type=str, help="Directory of the folder with data to be split (this folder should be a set of smaller folders)")
    parser.add_argument("--ratio", type=float, nargs="+", default=[0.8, 0.1, 0.1] , help="How dataset should be split training_ratio test_ratio val_ratio")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for randomly splitting data")

    args = parser.parse_args()

    main(args.image_dir, args.ratio, args.seed)