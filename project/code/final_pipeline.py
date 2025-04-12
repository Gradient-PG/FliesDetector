import os
from clearml import Task, PipelineController
from transfer_dataset import run_pipeline
# from notebooks.yolo_training import train_segmentator
# from notebooks.classifier_training import train_model

# Id of pipeline to load, split, and upload data into clearml
# ETL_PIPELINE_ID = "acc22a576068405ca5f75e73514b0ed4"

# etl_pipeline = Task.get_task(task_id=ETL_PIPELINE_ID)

# Main pipeline
# Structure:
#   - ETL Pipeline:
#       - Load data from external source
#       - Split data into train/val/test sets
#       - Upload data to ClearML Datesets
#   - Training Pipeline:
#       - Train Classifier
#       - Train Detector
#   - Measure Pipeline:
#       - Measure performance of Classifier
#       - Measure performance of Detector

pipeline = PipelineController(name="Complete Pipeline", project="Muszki", version="1.0")
pipeline.set_default_execution_queue("default")
# TODO od Huberta wziac wszystkie defaultowe url-e

        # label_studio_url=args.label_studio_url,
        # api_key=args.api_key,
        # export_type=args.export_type,
        # data_dir=args.data_dir,
        # ratio=args.ratio,
        # seed=args.seed,
        # clf_dataset_name=args.clf_dataset_name,
        # det_dataset_name=args.det_dataset_name

# pipeline.add_parameter(
#     name='label_studio_url',
#     description='Url to label studio, in order to pull the data', 
#     default="None"
# )
# pipeline.add_parameter(
#     name='api_key',
#     description='Api key to label studio', 
#     default="None"
# )
# pipeline.add_parameter(
#     name='export_type',
#     description='Format of data to pull from label studio', 
#     default="None"
# )

# Parameters

# ETL parameters
pipeline.add_parameter(
    name='split_ratio',
    description='How to split datasets', 
    default=(.8, .1, .1)
)
pipeline.add_parameter(
    name='seed',
    description='Nie wiem czy chcemy to miec', 
    default=42
)

# Detector training parameters
pipeline.add_parameter(
    name="detector_model",
    description="Name of supported detectors (For now only yolo11n)",
    default="yolo11n"
)
pipeline.add_parameter(
    name="detector_epochs",
    description="Number of epochs to training detector",
    default=32
)
pipeline.add_parameter(
    name="detector_batch_size",
    description="Batch size for training detector",
    default=8
)
# Classifier training parameters TODO poprosic zeby argumenty byly bardziej odkryte w kodzie, te ktore fajnie byloby miec
pipeline.add_parameter(
    name="classifier_model",
    description="Pretrained model from huggingface",
    default="google/vit-base-patch16-224"
)

etl_pipeline = Task.get_task(
    project_name="Muszki",
    task_name="Transfer Dataset",
    task_filter={'type': 'pipeline'}

)

# ETL Pipeline TODO dodac parametryzacje
pipeline.add_step(
    name="ETL-pipeline",
    base_task_id=etl_pipeline.id,
    parameter_override={
        "label_studio_url": 'http://localhost:8082',
        "api_key": "811180aeebfefc90700c48eef29231a6bedc6677", 
        "export_type": 'YOLO_WITH_IMAGES',
        "data_dir": f"{os.getcwd()}/project/data",
        "ratio": [0.8, 0.1, 0.1], 
        "seed": 42,
        "clf_dataset_name": 'test-classification-2', 
        "det_dataset_name": 'test-detection-2'
    }
)

# Object Detector Training Task
# pipeline.add_function_step(
#     name="Train-detector",
#     function=train_segmentator,
#     function_kwargs={
#         'dataset_id': "${ETL-pipeline.artifacts.Detection_Dataset}",
#         'project_name': "Muszki",
#         'task_name': "Detection",
#         'model_variant': "${pipeline.detector_model}",
#         'epochs': "${pipeline.detector_epochs}",
#         'batch_size': "${pipeline.detector_batch_size}"
#     },
#     parents=["ETL-pipeline"]
# )

# Classifier Detector Training Task
# pipeline.add_function_step(
#     name="Train-classifier",
#     function=train_model,
#     function_kwargs={
#     'dataset_id': "${ETL-pipeline.artifacts.Classification_Dataset}",
#     'model_checkpoint': "${pipeline.classifier_model}",
#     },
#     parents=["ETL-pipeline"]
# )

pipeline.start_locally()


# O co poprosic na spotkaniu
# Hubert:
#   - Aby dodał datasety, ktore tworzy jako artefakty w pipeline, abym mogl potem sie do nich odwolac bez problemu
# Kamil:
#   Aby zmienił lekko kod, tak aby przyjmował dataset juz podzielony na sety
# Wszystkich:
#   Jaki poziom parametryzacji chcemy zrobic, wedlug mnie najlepiej byloby zrobic .env w ktorym bylby url do label_studi i api_key etc.
#   A wszystkie takie luzne argumenty bylyby pobierane w trakcie wykonywania pipeline