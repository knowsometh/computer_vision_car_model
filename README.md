# RNN Classification Model Project - Computer Vision (CV)

This project builds a Recurrent Neural Network model for car model recognition.

## Project Overview

The purpose of this project is to train and evaluate an RNN model using Python.  
The project includes data preprocessing, model training, evaluation, and prediction.

## Technologies Used

- Python
- TensorFlow / PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Dataset

The full dataset is large and is not stored directly in this GitHub repository.

Full dataset source: https://www.kaggle.com/datasets/cyizhuo/stanford-cars-by-classes-folder

## Model Details & Performance
Model: Resnet50
Train: 80%
Test:20%
Optimiser: SGD
Framework: Pytorch

The model was evaluated on a validation set of 2,795 samples across 196 classes.

| Metric | Score |
|---|---:|
| Top-1 Accuracy | 70.7% |
| Top-5 Accuracy | 92.6% |
| Macro F1 Score | 70.0% |
| Micro F1 Score | 70.7% |

The Top-5 accuracy shows that the correct class appears within the model's top five predictions in approximately 92.6% of validation cases.