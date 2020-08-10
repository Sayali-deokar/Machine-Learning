import pandas as pd

milk = pd.read_csv("D:/Statistics (Python)/Datasets/milk.csv",
                   index_col=0)

from sklearn.preprocessing import StandardScaler
# Create scaler: scaler
scaler = StandardScaler()
milkscaled=scaler.fit_transform(milk)

milkscaled = pd.DataFrame(milkscaled,
                          columns=milk.columns,
                          index=milk.index)

# Import KMeans
from sklearn.cluster import KMeans

# Create a KMeans instance with clusters: model
model = KMeans(n_clusters=3,random_state=2019)

# Fit model to points
model.fit(milkscaled)

# Cluster Centroids
model.cluster_centers_

#model.n_init
# Determine the cluster labels of new_points: labels
labels = model.predict(milkscaled)

# Print cluster labels of new_points
print(labels)

clusterID = pd.DataFrame({'ClustID':labels})
clusteredData = pd.concat([milk.reset_index(),clusterID],axis=1)

### OR

clusterID = pd.DataFrame({'ClustID':labels},index=milk.index)
clusteredData = pd.concat([milk,clusterID],
                          axis='columns')

# Variation
print(model.inertia_)

clustNos = [2,3,4,5,6,7,8,9,10]
Inertia = []

for i in clustNos :
    model = KMeans(n_clusters=i,random_state=2019)
    model.fit(milkscaled)
    Inertia.append(model.inertia_)
    
# Import pyplot
import matplotlib.pyplot as plt

plt.plot(clustNos, Inertia, '-o')
plt.title("Scree Plot")
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(clustNos)
plt.show()

# Create a KMeans instance with clusters: Best k model
model = KMeans(n_clusters=4)

# Fit model to points
model.fit(milkscaled)
#model.n_init
# Determine the cluster labels of new_points: labels
labels = model.predict(milkscaled)

clusterID = pd.DataFrame(labels)
clusteredData = pd.concat([milk.reset_index(drop=True),clusterID],axis=1)
