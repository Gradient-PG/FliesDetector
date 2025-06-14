# FliesDetector

## Building and running the app

1. Prepare the data using script
```bash 
python project/code/scripts/ClassifierDataSplitter.py --image_dir project/data/label_studio
```
2. Train the model using script
```bash
python project/code/notebooks/classifier_training.py
```
3. Copy the model to the models dir
```bash
cp -R classifier_model/image_classification_model/* project/code/yolo2class_pipeline/models/classificator
```
4. Run docker containers
```bash
cd project
docker compose up
```
6. Open the app in your browser
```bash
http://localhost:7860
```
