import os
import random
import shutil
import yaml

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

    
    def split_files(self, train_part: float, test_part: float, val_part: float):
        # Randomly distribute file pairs in the list so that every split is different
        random.shuffle(self.file_pairs_list)

        total_count = len(self.file_pairs_list)

        train_count = int(total_count * train_part)
        test_count = int(total_count * test_part)
        val_count = int(total_count * val_part)

        # Create lists containing files for specified sets
        self.train_files = self.file_pairs_list[:train_count]
        self.test_files = self.file_pairs_list[train_count:train_count+test_count]
        self.val_files = self.file_pairs_list[train_count+test_count:train_count+test_count+val_count]

    def copy_files(self, file_pairs, dest_img, dest_lbl):
        # Copy specified files (train, val or test) to their destination directory
        for img, lbl in file_pairs:
            shutil.copy(img, dest_img)
            shutil.copy(lbl, dest_lbl)

    def create_yolo_yaml(self, filename='detection_set.yaml'):
        # Specify config of .yaml file for YOLO detector model
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

    def run(self, train_part=0.8, test_part=0.1, val_part=0.1):
        # This function is running whole pipeline for splitting data accordingly to rules
        self.create_dirs()
        self.collect_file_pairs()
        self.split_files(train_part, test_part, val_part)

        self.copy_files(self.train_files, self.DEST_PATHS['dest_img_train'], self.DEST_PATHS['dest_lbl_train'])
        self.copy_files(self.val_files, self.DEST_PATHS['dest_img_val'], self.DEST_PATHS['dest_lbl_val'])
        self.copy_files(self.test_files, self.DEST_PATHS['dest_img_test'], self.DEST_PATHS['dest_lbl_test'])

        self.create_yolo_yaml()

        print("Splitting completed.")

if __name__ == "__main__":
    source = '/home/kiterman/gradient/FliesDetector/project/data/label_studio'      
    destination = '/home/kiterman/gradient/FliesDetector/project/data/detection'    
    splitter = DetectorDataSplitter(source, destination)
    splitter.run()
        