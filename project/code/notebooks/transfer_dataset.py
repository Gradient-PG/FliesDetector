import argparse
import requests
import os
import zipfile
from PIL import Image

from label_studio_sdk.client import LabelStudio
import requests

from clearml import TaskTypes
from clearml.automation.controller import PipelineDecorator


def download_data(label_studio_url, api_key, export_type):
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
            with open(f'{project.id}.zip', 'wb') as f:
                f.write(response.content)
            with zipfile.ZipFile(f'{project.id}.zip', 'r') as zip_file:
                zip_file.extractall(f'{project.id}')
            os.remove(f'{project.id}.zip')
        except Exception:
            print(f'Downloading project with id={project.id} failed.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--label_studio_url', type=str, default='http://localhost:8082', help="URL where Label Studio is accessible")
    parser.add_argument('--api_key', type=str, help='API key for user account')
    parser.add_argument('--export_type', type=str, default='YOLO_WITH_IMAGES', help='Dataset export type, e.g. YOLO, COCO, etc.')
    
    args = parser.parse_args()

    download_data(args.label_studio_url, args.api_key, args.export_type)
