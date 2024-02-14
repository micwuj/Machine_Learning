import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Read data from TSV file
data = pd.read_csv('flats_for_clustering.tsv', header=0, sep='\t')

# Map floor levels to numerical values
data["Piętro"] = data["Piętro"].apply(lambda x: 0 if x in ["parter", "niski parter"] else x)
data["Piętro"] = data["Piętro"].apply(lambda x: 5 if x in ["poddasze"] else x)

# Filter data by removing rows where the price is greater than 1,000,000
data = data[data["cena"] <= 1000000]

data = data.dropna()

# Standardize the data using StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Number of clusters
n_clusters = 5

# Apply KMeans clustering to the scaled data
kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
kmeans.fit(scaled_data)

# Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

# Add PCA components and cluster labels
data[['X', 'y']] = principal_components
data['c'] = kmeans.labels_

plt.scatter(data['X'], data['y'], c=data['c'], cmap='viridis')
plt.title('Clusters Based on Two Principal Components')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()