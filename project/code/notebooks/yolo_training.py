from clearml import Task

from ultralytics import YOLO
import os
import torch


# Step 1: Creating a ClearML Task
task = Task.init(project_name="Muszki", task_name="Detection")

# Step 2: Selecting the YOLO11 Model
model_variant = "yolo11n"
task.set_parameter("model_variant", model_variant)

# Step 3: Loading the YOLO11 Model
model = YOLO(f"{model_variant}.pt")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 4: Setting Up Training Arguments
path_to_detection_yaml = os.path.join(os.getcwd(), 'project', 'data', 'detection', 'detection_set.yaml')
args = dict(
    data=path_to_detection_yaml, 
    epochs=32, 
    batch=8)
task.connect(args)

# Step 5: Initiating Model Training
results = model.train(**args)

print(results)

# model.eval()
model.val(data=path_to_detection_yaml, save_json=True)