# Spam Classification using Neural Network

This Python script demonstrates the implementation of a neural network for spam classification using text data. The dataset (`data.csv`) is loaded from a CSV file, preprocessed by removing 'Subject: ' from the text column, and split into training and testing sets. The script utilizes CountVectorizer for feature extraction, and a simple neural network is defined and trained using Keras and PyTorch.

## Dependencies
- pandas
- numpy
- scikit-learn
- keras
- PyTorch

## Usage
1. Ensure the required libraries are installed (`pip install pandas numpy scikit-learn keras`).
2. Provide the dataset (`data.csv`) in the same directory as the script.
3. Run the script with `python script_name.py`.

## Description - Keras
- **Data Loading and Preprocessing**: Reads data from a CSV file and removes 'Subject: ' from the text.
- **Feature Extraction**: Uses CountVectorizer to convert text data into a bag-of-words representation.
- **Neural Network Model**: Defines a simple neural network model using Keras for binary classification.
- **Training**: Trains the model on the training set using batch processing.
- **Evaluation**: Evaluates the model on the test set and prints accuracy, precision, recall, and F-score.

## Description - PyTorch
- **Data Loading and Preprocessing**: Reads data from a CSV file and removes 'Subject: ' from the text.
- **Neural Network Model**: Defines a simple neural network model using PyTorch for binary classification.
- **Training**: Trains the model on the training set using binary cross-entropy loss and Adam optimizer.
- **Evaluation**: Evaluates the model on the test set and prints accuracy, precision, recall, and F-score.

Note: Ensure the dataset (`data.csv`) follows the expected format and is present in the same directory as the script.