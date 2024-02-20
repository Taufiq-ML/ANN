# Combined Cycle Power Plant Energy Output Prediction using Artificial Neural Network (ANN)

This repository contains the code for building an Artificial Neural Network (ANN) regression model to predict the electrical energy output of a Combined Cycle Power Plant.

## Overview

The Combined Cycle Power Plant is a highly efficient system that generates electricity by combining gas and steam turbines. Predicting its energy output accurately is crucial for optimizing operations and energy management. This project leverages machine learning techniques, specifically ANN, to forecast the electrical energy output based on various input parameters.

## Dataset

The dataset used for this project is sourced from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant). It contains features such as temperature, pressure, and humidity, which influence the energy output. The target variable is the net hourly electrical energy output.

## Code Structure

The code is organized into three main parts:

### 1. Data Preprocessing

- The dataset is imported and split into input features (X) and target variable (y).
- Data preprocessing techniques such as normalization and train-test splitting are applied.

### 2. Building the ANN

- An ANN model is constructed using TensorFlow's Keras API.
- The architecture consists of multiple dense layers with rectified linear unit (ReLU) activation.
- The model is designed to learn the nonlinear relationships between input features and energy output.

### 3. Training and Evaluation

- The ANN model is compiled with appropriate loss function and optimizer.
- Training is performed on the training set with a specified batch size and number of epochs.
- The model's performance is evaluated on the test set using mean squared error (MSE) as the metric.
- Predictions are made on the test set, and results are compared with actual values.

## Usage

To run the code:

1. Clone this repository to your local machine.
2. Ensure you have the required libraries installed (NumPy, Pandas, TensorFlow).
3. Execute the code in a suitable Python environment such as Jupyter Notebook or Google Colab.

## Results

The trained ANN model demonstrates promising performance in predicting the electrical energy output of the Combined Cycle Power Plant. Evaluation metrics such as MSE can be used to assess the model's accuracy and fine-tune hyperparameters for further improvement.

## Contributors

Motivated by Passionate AI Instructor Hadelin de Ponteves and Kirill Eremenko (DS & AI Instructor) from UDEMY


## License

This project is licensed under the [MIT License](LICENSE).
