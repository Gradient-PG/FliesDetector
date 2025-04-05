import argparse
import requests
import os
import zipfile
from PIL import Image

from label_studio_sdk.client import LabelStudio
import requests

from clearml import TaskTypes
from clearml.automation.controller import PipelineDecorator

from scripts.data_splitter import DataSplitter
from scripts.classifier_data_splitter import ClassifierDatasetSplitter
from scripts.detector_data_splitter import DetectorDataSplitter


def download_data(label_studio_url, api_key, export_type, dataset_dir):
    '''
    Download data from label-studio
    '''

    # Get projects using label_studio_sdk
    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)
    projects = ls.projects.list()
    if not projects:
        raise Exception("No projects found")

    for project in projects:
        headers = {'Authorization': f"Token {api_key}"}
        url = f'{label_studio_url}/api/projects/{project.id}/export?exportType={export_type}'
        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                raise Exception
            # TODO it only works for formats which return zip
            # Download, extract and remove zip file

            dataset_path = os.path.join(dataset_dir, str(project.id))
            with open(f'{dataset_path}.zip', 'wb') as f:
                f.write(response.content)
            with zipfile.ZipFile(f'{dataset_path}.zip', 'r') as zip_file:
                zip_file.extractall(f'{dataset_path}')
            os.remove(f'{dataset_path}.zip')
            print(f'Project with id={project.id} downloaded and extracted successfully.')
        except Exception:
            print(f'Downloading project with id={project.id} failed.')

def process_data(data_dir, ratio, seed):
    ClassifierDatasetSplitter(data_dir, tuple(ratio), seed)()
    DetectorDataSplitter(data_dir, tuple(ratio), seed)()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--label_studio_url', type=str, default='http://localhost:8082', help="URL where Label Studio is accessible")
    parser.add_argument('--api_key', type=str, help='API key for user account')
    parser.add_argument('--export_type', type=str, default='YOLO_WITH_IMAGES', help='Dataset export type, e.g. YOLO, COCO, etc.')
    parser.add_argument("--data_dir", type=str, default=f"{os.getcwd()}/project/data", help="Directory of the root folder with data")
    parser.add_argument("--ratio", type=float, nargs="+", default=[0.8, 0.1, 0.1] , help="How dataset should be split training_ratio test_ratio val_ratio")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for randomly splitting data")
    
    args = parser.parse_args()

    download_data(args.label_studio_url, args.api_key, args.export_type, os.path.join(args.data_dir, 'label_studio'))
    process_data(args.data_dir, args.ratio, args.seed)