from datasets import DatasetDict, load_dataset
from clearml import Task, Dataset, Logger, PipelineDecorator
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
import torch
import evaluate
import numpy as np
from collections import Counter

# Main training pipeline function
@PipelineDecorator.component()
def train_model(
    dataset_id="fbaaa3930a40462482043d1cd6963096",
    model_checkpoint="google/vit-base-patch16-224",
    output_path="classifier_model",
    training_args=None,
    split_names=None 
):
    # Custom callback to log metrics and upload final model to ClearML
    class ClearMLCallback(TrainerCallback):
        def __init__(self, task):
            self.task = task
            self.best_accuracy = 0
            
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_world_process_zero and logs:
                for key, value in logs.items():
                    Logger.current_logger().report_scalar(
                        title="Training Metrics",
                        series=key,
                        value=value,
                        iteration=state.global_step
                    )

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if state.is_world_process_zero and metrics:
                for key, value in metrics.items():
                    Logger.current_logger().report_scalar(
                        title="Validation Metrics",
                        series=key,
                        value=value,
                        iteration=state.global_step
                    )
                current_acc = metrics.get("eval_accuracy", 0)
                if current_acc > self.best_accuracy:
                    self.best_accuracy = current_acc
                    Logger.current_logger().report_single_value(
                        "Best Accuracy", self.best_accuracy
                    )

        def on_train_end(self, args, state, control, **kwargs):
            self.task.update_output_model(
                model_path="image_classification_model",
                model_name="ViT Classifier",
                auto_delete_file=False
            )

    # Load dataset from ClearML by ID
    def get_clearml_dataset(dataset_id):
        clearml_dataset = Dataset.get(dataset_id=dataset_id)
        return clearml_dataset.get_local_copy()

    # Prepare dataset: assume train/val/test are already split in folders
    def prepare_datasets(dataset_path, feature_extractor,split_names=None):
        dataset = load_dataset("imagefolder", data_dir=dataset_path)
        if split_names is None:
            split_names = {
                "train": "train",
                "validation": "validation",
                "test": "test"
            }
        # Ensure required splits are present
        expected_keys = set(split_names.values())
        if not expected_keys.issubset(dataset.keys()):
            raise ValueError(f"Dataset must contain splits: {expected_keys}. Found: {set(dataset.keys())}")

        # Apply feature extraction and label assignment
        def transform(example_batch):
            inputs = feature_extractor(
                [image.convert("RGB") for image in example_batch["image"]],
                return_tensors="pt"
            )
            inputs["labels"] = example_batch["label"]
            return inputs

        # Preprocess the datasets
        preprocessed_dataset = DatasetDict({
            split: dataset[original_name].map(
                transform,
                batched=True,
                batch_size=32,
                remove_columns=dataset[original_name].column_names
            )
            for split, original_name in split_names.items()
        })
        return preprocessed_dataset, dataset[split_names["train"]].features["label"].names

    # Log dataset statistics to ClearML
    def log_dataset_stats(dataset, labels, task):
        stats = {
            "train_samples": len(dataset["train"]),
            "valid_samples": len(dataset["validation"]),
            "test_samples": len(dataset["test"]),
            "class_distribution": {
                labels[i]: count for i, count in 
                Counter(dataset["train"]["labels"]).most_common()
            }
        }
        task.connect(stats, name="Dataset Statistics")
        print("Dataset stats:", stats)
    # ClearML task initialization
    task = Task.init(
        project_name="Muszki",
        task_name="Classification",
        tags=["huggingface", "classification", "pipeline"]
    )
    
    # Define default training arguments if not passed
    if training_args is None:
        training_args = {
            "output_dir": f"{output_path}/results",
            "evaluation_strategy": "epoch",
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 32,
            "num_train_epochs":10,
            "learning_rate": 5e-4,
            "warmup_steps": 500,
            "logging_steps": 10,
            "remove_unused_columns": False,
            "fp16": False,
            "dataloader_num_workers": 4,
            "weight_decay": 0.01,
            "report_to": "none"
        }

    task.connect(training_args)
    training_args = TrainingArguments(**training_args)

    # Load and prepare datasets
    dataset_path = get_clearml_dataset(dataset_id)
    feature_extractor = AutoImageProcessor.from_pretrained(model_checkpoint)
    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        ignore_mismatched_sizes=True
    )
    preprocessed_dataset, labels = prepare_datasets(dataset_path, feature_extractor, split_names)

    log_dataset_stats(preprocessed_dataset, labels, task)
    # stats = {
    #     "train_samples": len(preprocessed_dataset["train"]),
    #     "valid_samples": len(preprocessed_dataset["validation"]),
    #     "test_samples": len(preprocessed_dataset["test"]),
    #     "class_distribution": {
    #         labels[i]: count for i, count in 
    #         Counter(preprocessed_dataset["train"]["labels"]).most_common()
    #     }
    # }
    # task.connect(stats, name="Dataset Statistics")
    # print("Dataset stats:", stats)

    # Define accuracy metric
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=preprocessed_dataset["train"],
        eval_dataset=preprocessed_dataset["validation"],
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
        callbacks=[ClearMLCallback(task)]
    )

    # Training and validation
    trainer.train()
    eval_results = trainer.evaluate()
    task.get_logger().report_single_value("Final Accuracy", eval_results["eval_accuracy"])

    # Final test set evaluation
    test_results = trainer.evaluate(eval_dataset=preprocessed_dataset["test"])
    task.get_logger().report_scalar(
        title="Test Metrics",
        series="Test Accuracy",
        value=test_results["eval_accuracy"],
        iteration=0
    )

    # Save model and processor
    model.save_pretrained(f"{output_path}/model")
    feature_extractor.save_pretrained(f"{output_path}/feature_extractor")

    # Upload model to ClearML
    task.update_output_model(
        model_path=f"{output_path}/model",
        model_name="ViT Classifier",
        auto_delete_file=False
    )

    return {
        "model_path": f"{output_path}/model",
        "feature_extractor_path": f"{output_path}/feature_extractor",
        "eval_results": eval_results,
        "test_results": test_results
    }


if __name__ == "__main__":
    
    result = train_model()
    print("Results:", result)