# Linear Regression and Ridge Regression

This Python script demonstrates the use of Linear Regression and Ridge Regression on a dataset. The dataset (`communities.data`) is loaded using pandas, preprocessed, and split into training and testing sets. The features are standardized using StandardScaler. The script then trains a Linear Regression model without regularization and a Ridge Regression model with regularization (alpha=1.0).

## Dependencies
- pandas
- scikit-learn

## Usage
1. Ensure the required libraries are installed (`pip install pandas scikit-learn`).
2. Run the script with `python main.py`.

## Description
- `Linear Regression`: Fits a linear model to predict the target variable.
- `Ridge Regression`: Linear Regression with L2 regularization to prevent overfitting.

The Root Mean Squared Error (RMSE) is calculated for both models to evaluate performance.

Note: Ensure the dataset (`communities.data`) is present in the same directory as the script.
