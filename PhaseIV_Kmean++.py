# K-mean++
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


data = pd.read_csv("realtor-data-cleaned3.csv")
X = data[['bed', 'acre_lot', 'zip_code', 'house_size', 'city']]
y = data['price']

range_n_clusters = list(range(2, 11))
silhouette_scores = []

for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)

plt.plot(range_n_clusters, silhouette_scores, marker='o')
plt.title('Silhouette Scores for Different Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid()
plt.show()

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

scaled_wcss = [x / 1e6 for x in wcss]

plt.plot(range(1, 11), scaled_wcss, marker='o')
plt.title('K-Means Clustering: Within-Cluster Sum of Squares (in millions)')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (in millions)')
plt.grid()
plt.show()