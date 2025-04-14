from clearml import Task, Dataset, PipelineDecorator
from ultralytics import YOLO
import os
import torch
import yaml  

@PipelineDecorator.component(return_values=["results"])
def train_segmentator(
    dataset_id: str,
    project_name: str = "Muszki",
    task_name: str = "Detection",
    model_variant: str = "yolo11n",
    epochs: int = 32,
    batch_size: int = 8
):
    """Training function integrating ClearML with YOLO"""
    try:
        # Loading dataset from ClearML
        dataset = Dataset.get(dataset_id=dataset_id)
        dataset_path = dataset.get_local_copy()
        
        # Generating a dynamic YAML file
        yaml_config = {
            'names': ['fly'], 
            'nc': 1,           
            'path': dataset_path,
            'train': 'images/train', 
            'val': 'images/val',
            'test': 'images/test'      
        }

        # Save YAML file
        yaml_path = os.path.join(dataset_path, 'detection_set.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)

        # Loading Model
        model = YOLO(f"{model_variant}.pt")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        args = dict(
            data=yaml_path,
            epochs=epochs,
            batch=batch_size,
            device=device.index if device.type == 'cuda' else 'cpu'
        )

        # Trening
        results = model.train(**args)

        # Validation
        # model.val(data=yaml_path, save_json=True)

    except Exception as e:
        print(f"Błąd podczas treningu: {str(e)}")
        raise

if __name__ == "__main__":
    train_segmentator(
        dataset_id="ab637d3b70144a949f68dac5baac4d0f",  
        model_variant="yolo11n",
        epochs=32,
        batch_size=8
    )

