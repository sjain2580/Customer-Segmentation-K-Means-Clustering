# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from warnings import simplefilter

print("Step 1: Libraries Imported.")

# Step 2: Load the Dataset
# The Mall Customer Segmentation dataset is publicly available on Kaggle.
try:
    url = 'Mall_Customers.csv'
    df = pd.read_csv(url)
    print("Step 2: Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Step 3: Data Preprocessing
print("Step 3: Data Preprocessing and Feature Selection.")

# Select features for clustering: Annual Income and Spending Score.
# These are key behavioral features for customer segmentation.
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale the data. This is crucial for distance-based algorithms like K-Means.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Data scaled and ready for clustering.")

# Step 4: Use the Elbow Method to find the optimal number of clusters (k)
print("\nStep 4: Running Elbow Method to find the optimal 'k'...")
inertia = []
# Test a range of possible cluster numbers (k=1 to k=10)
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid(True)
plt.savefig('Elbow_curve.png')
plt.show()

print("Elbow plot generated. Look for the 'elbow' to determine the optimal k.")

# Step 5: Train the K-Means model with the optimal k
# From the plot, we'll assume the optimal k is 5 (a common result for this dataset).
optimal_k = 5
print(f"\nStep 5: Training K-Means model with optimal k = {optimal_k}...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("K-Means model trained and clusters assigned to the dataset.")

# Step 6: Evaluate the model
print("\nStep 6: Evaluating the model using Silhouette Score...")
score = silhouette_score(X_scaled, df['Cluster'])
print(f"Silhouette Score: {score:.4f}")

# Step 7: Visualization of Clusters
print("\nStep 7: Visualizing the clusters.")

# Plot the scatter plot of the clusters
plt.figure(figsize=(10, 8))
scatter = sns.scatterplot(
    data=df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='viridis',
    s=100,
    alpha=0.7,
    edgecolor='k'
)

# Plot the cluster centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    x=centroids[:, 0],
    y=centroids[:, 1],
    marker='X',
    s=200,
    c='red',
    edgecolor='k',
    label='Centroids'
)

plt.title('Customer Segments based on Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.grid(True)
plt.savefig('cluster.png')
plt.show()
