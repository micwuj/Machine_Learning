# Flat Clustering using KMeans

This Python script demonstrates the use of KMeans clustering on a dataset of flats. The dataset (`flats_for_clustering.tsv`) is loaded using pandas, preprocessed by mapping floor levels to numerical values and filtering by price. The features are standardized using StandardScaler, and KMeans clustering is applied. The script also uses Principal Component Analysis (PCA) to visualize the clusters in two dimensions.

## Dependencies
- pandas
- scikit-learn
- matplotlib

## Usage
1. Ensure the required libraries are installed (`pip install pandas scikit-learn matplotlib`).
2. Run the script with `python main.py`.

## Description
- `KMeans Clustering`: Groups flats into clusters based on their features.
- `Principal Component Analysis (PCA)`: Reduces dimensionality for visualization.
- Scatter plot displays clusters in two principal components.

Note: Ensure the dataset (`flats_for_clustering.tsv`) is present in the same directory as the script.
