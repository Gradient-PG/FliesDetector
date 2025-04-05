from clearml import Task

from ultralytics import YOLO
import os
import torch


task = Task.init(project_name="Muszki", task_name="Detection")

model_variant = "yolov8n"
task.set_parameter("model_variant", model_variant)

model = YOLO(f"{model_variant}.pt")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

path_to_detection_yaml = os.path.join(os.getcwd(), 'project', 'data', 'detection', 'detection_set.yaml')
args = dict(
    data=path_to_detection_yaml, 
    epochs=32, 
    batch=8)
task.connect(args)

results = model.train(**args)

print(results)

model.val(data=path_to_detection_yaml, save_json=True)
