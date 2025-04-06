from datasets import DatasetDict, load_dataset
from clearml import Task, Dataset, Logger 
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

class ClearMLCallback(TrainerCallback):
    def __init__(self,task):
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
            # Log validation metrics
            for key, value in metrics.items():
                Logger.current_logger().report_scalar(
                    title="Validation Metrics",
                    series=key,
                    value=value,
                    iteration=state.global_step
                )
            
            # Track best accuracy
            current_acc = metrics.get("eval_accuracy", 0)
            if current_acc > self.best_accuracy:
                self.best_accuracy = current_acc
                Logger.current_logger().report_single_value(
                    "Best Accuracy", self.best_accuracy
                )

    def on_train_end(self, args, state, control, **kwargs):
        # Final model upload
        self.task.update_output_model(
            model_path="image_classification_model",
            model_name="ViT Classifier",
            auto_delete_file=False
        )

def get_clearml_dataset(dataset_id):
    clearml_dataset = Dataset.get(dataset_id=dataset_id)
    return clearml_dataset.get_local_copy()

def prepare_datasets(dataset_path, feature_extractor):
    dataset = load_dataset("imagefolder", data_dir=dataset_path)
    
    if list(dataset.keys()) == ['train']:
        split = dataset["train"].train_test_split(test_size=0.2, seed=42)
        dataset = DatasetDict({
            "train": split["train"],
            "validation": split["test"]  
        })
    required_splits = {'train', 'validation'}
    if not required_splits.issubset(dataset.keys()):
        raise ValueError(f"Dataset must contain {required_splits}. accessible splits: {list(dataset.keys())}")
    # Preprocessing function
    def transform(example_batch):
        # Process images with feature_extractor
        inputs = feature_extractor(
            [image.convert("RGB") for image in example_batch["image"]],
            return_tensors="pt"
        )
        inputs["labels"] = example_batch["label"]
        return inputs
    # Apply preprocessing
    preprocessed_dataset = dataset.map(
        transform,
        batched=True,
        batch_size=32,
        remove_columns=dataset["train"].column_names # Remove raw images to save memory
    )
    
    return preprocessed_dataset, dataset["train"].features["label"].names

def log_dataset_stats(dataset, labels, task):
    stats = {
        "train_samples": len(dataset["train"]),
        "valid_samples": len(dataset["validation"]),
        "class_distribution": {
            labels[i]: count for i, count in 
            Counter(dataset["train"]["labels"]).most_common()
        }
    }
    task.connect(stats, name="Dataset Statistics")
    print("Dataset stats:", stats)

def train_model(
    dataset_id="c2a05425242448ab84040a8cbaa6e639",
    model_checkpoint="google/vit-base-patch16-224",
    output_path="classifier_model",
    training_args=None
):
    """Main training function for use in pipelines"""
    
    # Task initialization in clearml
    task = Task.init(
        project_name="Muszki",
        task_name="Classification",
        tags=["huggingface", "classification", "pipeline"]
    )
    
    # Training configuration
    if training_args is None:
        training_args = {
            "output_dir": f"{output_path}/results",
            "evaluation_strategy": "epoch",
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 32,
            "num_train_epochs": 4,
            "learning_rate": 5e-4, # Increased learning rate
            "warmup_steps": 500,  # Added warmup
            "logging_steps": 10,
            "remove_unused_columns": False,
            "fp16": False,   # Disabled mixed precision for debugging
            "dataloader_num_workers": 4,
            "weight_decay": 0.01,
            "report_to": "none"
        }
    
    task.connect(training_args)
    training_args = TrainingArguments(**training_args)
    
    # Loading data
    dataset_path = get_clearml_dataset(dataset_id)
    
    # Model preparation
    feature_extractor = AutoImageProcessor.from_pretrained(model_checkpoint)
    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        ignore_mismatched_sizes=True
    )
    
    # Data preparation
    preprocessed_dataset, labels = prepare_datasets(dataset_path, feature_extractor)
    log_dataset_stats(preprocessed_dataset, labels, task)
    
    # Metrics
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=preprocessed_dataset["train"],
        eval_dataset=preprocessed_dataset["validation"],
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
        callbacks=[ClearMLCallback(task)]  
    )
    trainer.train()
    eval_results = trainer.evaluate()
    task.get_logger().report_single_value("Final Accuracy", eval_results["eval_accuracy"])
    
    # Save model
    model.save_pretrained(f"{output_path}/model")
    feature_extractor.save_pretrained(f"{output_path}/feature_extractor")
    
    # Upload artifacts
    task.update_output_model(
        model_path=f"{output_path}/model",
        model_name="ViT Classifier",
        auto_delete_file=False
    )
    
    return {
        "model_path": f"{output_path}/model",
        "feature_extractor_path": f"{output_path}/feature_extractor",
        "eval_results": eval_results
    }

if __name__ == "__main__":
    result = train_model()
    
    print("Results:", result)