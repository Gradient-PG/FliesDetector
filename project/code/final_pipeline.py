import os
from clearml import PipelineDecorator, Task, PipelineController
from transfer_dataset import download_data, process_data, upload_data
from notebooks.yolo_training import train_segmentator
from notebooks.classifier_training import train_model
from dotenv import load_dotenv


@PipelineDecorator.pipeline(name="Complete Pipeline", project="Muszki", version="1.0")
def full_pipeline(label_studio_url, api_key, export_type, data_dir, ratio, seed, clf_dataset_name, det_dataset_name, 
                  detector_model_variant, detector_epochs, detector_batch_size, 
                  classifier_model_checkpoint, classifier_output_path):
    # Perform dataset preparation
    data_dir = download_data(label_studio_url, api_key, export_type, data_dir)
    print("Finished downloading")
    data_dir = process_data(data_dir, ratio, seed)
    print("Finished processing")
    classifier_dataset, detection_dataset = upload_data(data_dir, clf_dataset_name, det_dataset_name)
    print("Finished uploading")
    train_segmentator(
        dataset_id=detection_dataset,
        model_variant=detector_model_variant,
        epochs=detector_epochs,
        batch_size=detector_batch_size
    )
    train_model(
        dataset_id=classifier_dataset,
        model_checkpoint=classifier_model_checkpoint,
        output_path=classifier_output_path
    )
    return 0

if __name__ == "__main__":
    load_dotenv(os.path.join(os.getcwd(), ".env"))
    PipelineDecorator.run_locally()
    full_pipeline(
        label_studio_url=os.getenv("LABEL_STUDIO_URL"),
        api_key=os.getenv("API_KEY"),
        export_type='YOLO_WITH_IMAGES',
        data_dir=f"{os.getcwd()}/project/data",
        ratio=[0.8, 0.1, 0.1],
        seed=42,
        clf_dataset_name='test-classification-2',
        det_dataset_name='test-detection-2',
        detector_model_variant="yolo11n",
        detector_epochs=32,
        detector_batch_size=8,
        classifier_model_checkpoint="google/vit-base-patch16-224",
        classifier_output_path="classifier_model"
    )