from clearml import Task, Dataset
from ultralytics import YOLO
import os
import torch
import yaml  

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
        # Creating a ClearML Task
        task = Task.init(project_name=project_name, task_name=task_name)
        
        # Loading dataset from ClearML
        dataset = Dataset.get(dataset_id=dataset_id)
        dataset_path = dataset.get_local_copy()
        
        # Generating a dynamic YAML file
        yaml_config = {
            'names': ['fly'], 
            'nc': 1,           
            'path': dataset_path,
            'train': 'train', 
            'val': 'val',
            'test': 'test'      
        }

        # Save YAML file
        yaml_path = os.path.join(dataset_path, 'detection_set.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)  
        
        # Setting Up Training Arguments
        task.set_parameters({
            "model_variant": model_variant,
            "epochs": epochs,
            "batch_size": batch_size,
            "dataset_id": dataset_id,
            "dataset_path": dataset_path,
            "yaml_path": yaml_path
        })

        # Loading Model
        model = YOLO(f"{model_variant}.pt")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Trening
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            batch=batch_size,
            device=device.index if device.type == 'cuda' else 'cpu'
        )

        # Validation
        model.val(data=yaml_path, save_json=True)
        
        return results

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_segmentator(
        dataset_id="ab637d3b70144a949f68dac5baac4d0f",  
        model_variant="yolo11n",
        epochs=2,
        batch_size=8
    )

