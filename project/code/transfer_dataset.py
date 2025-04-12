import argparse
import os
from clearml.automation.controller import PipelineDecorator


@PipelineDecorator.component()
def download_data(label_studio_url, api_key, export_type, dataset_dir):
    '''
    Download data from label-studio
    '''

    import os
    import requests
    import zipfile
    from label_studio_sdk.client import LabelStudio

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
            dataset_path = os.path.join(dataset_dir, 'label_studio', str(project.id))
            with open(f'{dataset_path}.zip', 'wb') as f:
                f.write(response.content)
            with zipfile.ZipFile(f'{dataset_path}.zip', 'r') as zip_file:
                zip_file.extractall(f'{dataset_path}')
            os.remove(f'{dataset_path}.zip')
            print(f'Project with id={project.id} downloaded and extracted successfully.')
        except Exception:
            print(f'Downloading project with id={project.id} failed.')

    return dataset_dir

@PipelineDecorator.component()
def process_data(data_dir, ratio, seed):
    '''
    Process data for classification and detection
    '''

    from scripts.classifier_data_splitter import ClassifierDatasetSplitter
    from scripts.detector_data_splitter import DetectorDataSplitter

    ClassifierDatasetSplitter(data_dir, tuple(ratio), seed)()
    print('Classifier dataset split successfully.')

    DetectorDataSplitter(data_dir, tuple(ratio), seed)()
    print('Detector dataset split successfully.')

    return data_dir

@PipelineDecorator.component()
def upload_data(data_dir, clf_dataset_name, det_dataset_name):
    '''
    Upload data to clearml
    '''

    from clearml import Dataset, Task

    task = Task.current_task()

    classification_dataset = Dataset.create(dataset_name=clf_dataset_name, dataset_project='Muszki')
    classification_dataset.add_files(path=os.path.join(data_dir, 'classification'))
    task.upload_artifact(name="Classification_Dataset", artefact_object=classification_dataset.id)
    print('Classification dataset uploaded successfully.')

    detection_dataset = Dataset.create(dataset_name=det_dataset_name, dataset_project='Muszki')
    detection_dataset.add_files(path=os.path.join(data_dir, 'detection'))
    task.upload_artifact(name="Detection_Dataset", artefact_object=detection_dataset.id)
    print('Detection dataset uploaded successfully.')

@PipelineDecorator.pipeline(name='Transfer Dataset', project='Muszki', version='1.0')
def run_pipeline(label_studio_url, api_key, export_type, data_dir, ratio, seed, clf_dataset_name, det_dataset_name):
    '''
    Pipeline to download, process and upload data
    '''

    #TODO Components have to return argument to next step so that they wait for each other
    data_dir = download_data(label_studio_url, api_key, export_type, data_dir)
    data_dir = process_data(data_dir, ratio, seed)
    upload_data(data_dir, clf_dataset_name, det_dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--label_studio_url', type=str, default='http://localhost:8082', help="URL where Label Studio is accessible")
    parser.add_argument('--api_key', type=str, help='API key for user account')
    parser.add_argument('--export_type', type=str, default='YOLO_WITH_IMAGES', help='Dataset export type, e.g. YOLO, COCO, etc.')
    parser.add_argument("--data_dir", type=str, default=f"{os.getcwd()}/project/data", help="Directory of the root folder with data")
    parser.add_argument("--ratio", type=float, nargs="+", default=[0.8, 0.1, 0.1] , help="How dataset should be split training_ratio test_ratio val_ratio")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for randomly splitting data")
    parser.add_argument("--clf_dataset_name", type=str, default='test-classification-2', help="Name of the classification dataset in clearml")
    parser.add_argument("--det_dataset_name", type=str, default='test-detection-2', help="Name of the detection dataset in clearml")
    
    args = parser.parse_args()

    PipelineDecorator.run_locally()
    run_pipeline(
        label_studio_url=args.label_studio_url,
        api_key=args.api_key,
        export_type=args.export_type,
        data_dir=args.data_dir,
        ratio=args.ratio,
        seed=args.seed,
        clf_dataset_name=args.clf_dataset_name,
        det_dataset_name=args.det_dataset_name
    )