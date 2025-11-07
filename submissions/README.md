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




## TASK 2

  # Clock Time Regression Models

## Description
This project implements CNN regression models to predict time from clock images using two different approaches:
- **Plain Regression**: Directly predicts time as a continuous value
- **Periodic Regression**: Uses sine/cosine encoding to handle the circular nature of time

## Requirements
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn

## Installation
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Data Setup
The code expects data files in the following structure:
```
data/A1_data_150/
    images.npy
    labels.npy
```

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
