from sklearn.cluster import KMeans
import numpy as np

# Data: [Annual Income (k$), Spending Score (1-100)]
X = np.array([[15, 39], [16, 81], [17, 6], [18, 77], [70, 40], [72, 45], [80, 15], [85, 12]])

# 3 Clusters: Low, Mid, and High spending/income levels
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

print("Customer Segmentation Results:")
for i, group in enumerate(clusters):
    print(f"Customer {i+1}: Cluster {group}")
