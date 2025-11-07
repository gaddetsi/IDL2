# Linux is required for GPU usage!

## TASK 1 
# MLP and CNN Image Classification

## Description
This project implements and compares Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN) models for image classification on Fashion-MNIST and CIFAR-10 datasets.

## Requirements
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

## Installation
```bash
pip install tensorflow numpy matplotlib
```

## Usage
Run the script directly:
```bash
python task1_mlp_cnn.py
```


## Output
The script provides:
- Model architecture summaries
- Training progress for each model
- Test accuracy results
- Performance comparison between MLP and CNN models


# TASK 2

### Requirements
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn

### Installation
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

### Data Setup
The code expects data files in the following structure:
```
data/A1_data_150/
    images.npy
    labels.npy
```

## TASK 2.1.a

  ### Categorical Model with different loss functions

## Description
This project implements CNN classification models to predict time from a clock image.
The following loss functions are compared using 24 classes:
- **Commonsense MSE**: Custom loss function combining MSE with common sense
- **Commonsense MSE 0**: Custom loss function combining MSE with common sense without feedback to the correct class
- **MSE**: Regular MSE
- **Categorical cross-entropy**: Regular Categorical cross-entropy
Afterwards Commonsense MSE is used for 720 classes.

## Usage
Run the script directly, get metrics and make plots:
```bash
python 2.1.a.py
python ./create_plots_cat.py ./saved_models/loss_*
```


## TASK 2.1.b

  # Clock Time Regression Models

## Description
This project implements CNN regression models to predict time from clock images using two different approaches:
- **Plain Regression**: Directly predicts time as a continuous value
- **Periodic Regression**: Uses sine/cosine encoding to handle the circular nature of time

## Usage
Run the script directly:
```bash
python regression_model.py
```

## What the Code Does
1. Loads and preprocesses clock image data
2. Implements two CNN models for time prediction
3. Trains both models and evaluates performance
4. Generates error analysis and comparison plots
5. Saves results in an `images/` directory

## Output
- Model performance metrics (mean error, accuracy within thresholds)
- Error distribution histograms
- Prediction vs true value scatter plots
- Training history comparison
- Automatic saving of the best periodic model

The script will print detailed performance comparisons between the two approaches.
