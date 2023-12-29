import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Membaca dataset dari file CSV
df = pd.read_csv('C:\Customers.csv')

df = pd.DataFrame(df)

# Menentukan jumlah kelompok (clusters)
k = 3

# Memilih kolom yang akan digunakan untuk clustering
selected_columns = ['Age', 'Annual Income ($)', 'Spending Score (1-100)']
selected_data = df[selected_columns]

# Membuat objek KMeans dengan jumlah kelompok k
kmeans = KMeans(n_clusters=k)

# Melatih model KMeans dengan data
kmeans.fit(selected_data)

# Mendapatkan label kelompok untuk setiap data point
labels = kmeans.labels_

# Menambahkan kolom label ke dalam DataFrame
df['Cluster'] = labels

# Visualisasi hasil dalam bentuk scatter plot 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b']
for i in range(k):
    cluster_data = selected_data[df['Cluster'] == i]
    ax.scatter(cluster_data['Age'], cluster_data['Annual Income ($)'], cluster_data['Spending Score (1-100)'],
               c=colors[i], label=f'Cluster {i + 1}')

ax.set_xlabel('Age')
ax.set_ylabel('Annual Income ($)')
ax.set_zlabel('Spending Score (1-100)')
ax.set_title('K-Means Clustering')
ax.legend()
plt.show()
