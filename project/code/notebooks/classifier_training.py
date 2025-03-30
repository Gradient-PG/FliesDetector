from datasets import load_dataset
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
from clearml import Task, Dataset, Logger
from collections import Counter
from sklearn.metrics import confusion_matrix

MODEL_OUTPUT_PATH = "classifier_model"

task = Task.init(
    project_name="Muszki",
    task_name="Classification",
    tags=["huggingface", "classification"]
)

# clearml_dataset = Dataset.get(
#     dataset_project="Datasets",
#     dataset_name="MyClassificationData"
# )
# dataset_path = clearml_dataset.get_local_copy()
dataset_path = "project/data/classification"

# Load dataset
dataset = load_dataset("imagefolder", data_dir=dataset_path)

# Check label names
labels = dataset["train"].features["label"].names
num_labels = len(labels)
print(f"Number of classes: {num_labels}")
print(f"Class names: {labels}")

# Load pretrained feature extractor and model
model_checkpoint = "google/vit-base-patch16-224"
feature_extractor = AutoImageProcessor.from_pretrained(model_checkpoint)

class ClearMLCallback(TrainerCallback):
    def __init__(self):
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
        task.update_output_model(
            model_path="image_classification_model",
            model_name="ViT Classifier",
            auto_delete_file=False
        )

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
    remove_columns=dataset["train"].column_names  # Remove raw images to save memory
)

# Load model
model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    ignore_mismatched_sizes=True  # In case you're changing number of classes
)

# Training configuration
training_args = TrainingArguments(
    output_dir=f"{MODEL_OUTPUT_PATH}/results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    learning_rate=5e-4,  # Increased learning rate
    warmup_steps=500,  # Added warmup
    logging_steps=10,
    remove_unused_columns=False,
    fp16=False,  # Disabled mixed precision for debugging
    dataloader_num_workers=4,
    gradient_accumulation_steps=1,
    weight_decay=0.01,
    report_to="none"
)

def log_dataset_stats():
    print(Counter(dataset["train"]["label"]).most_common())
    stats = {
        "train_samples": len(dataset["train"]),
        "valid_samples": len(dataset["test"]),
        "class_distribution": {
            labels[i]: count for i, count in 
            Counter(dataset["train"]["label"]).most_common()
        }
    }
    task.connect(stats, name="Dataset Statistics")

log_dataset_stats()

task.connect(training_args.to_dict())  # Log all training arguments
task.connect({"model_name": model_checkpoint})

# Metric configuration
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
    eval_dataset=preprocessed_dataset["test"],  # or ["validation"] if available
    compute_metrics=compute_metrics,
    tokenizer=feature_extractor,  # For image feature collation
    callbacks=[ClearMLCallback()] 
)

with open(f"{MODEL_OUTPUT_PATH}/model_arch.txt", "w") as f:
    print(model, file=f)
task.upload_artifact("Model Architecture", f"{MODEL_OUTPUT_PATH}/model_arch.txt")

# Start training
train_results = trainer.train()

# Evaluate after training
eval_results = trainer.evaluate()
print(f"Final evaluation results: {eval_results}")

# Save model and feature extractor
model.save_pretrained(f"{MODEL_OUTPUT_PATH}/image_classification_model")
feature_extractor.save_pretrained(f"{MODEL_OUTPUT_PATH}/image_classification_feature_extractor")

task.upload_artifact("Final Model", f"{MODEL_OUTPUT_PATH}/image_classification_model")
task.upload_artifact("Feature Extractor", f"{MODEL_OUTPUT_PATH}/image_classification_feature_extractor")

task.close()