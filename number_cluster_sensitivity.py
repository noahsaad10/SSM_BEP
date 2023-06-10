from stl import mesh
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

mesh_name = r'C:\Users\noah-\Desktop\TU\Year 3\BEP\COW\groomed\574_groomed.stl'
#mesh_name = r"C:\Users\noah-\Desktop\TU\Year 3\BEP\Geometries\Pelvis\TrajectoryVolume_9_Superior Ramus Right.stl"

mesh_file = mesh.Mesh.from_file(mesh_name)
_, com, _ = mesh_file.get_mass_properties()
mesh_file.translate(-com)
# Convert the mesh to a numpy array
points = np.asarray(mesh_file.vectors.reshape(-1, 3))

kmeans_kwargs = {"n_init": 8, "random_state": 0}

# A list holds the SSE values for each k and silhouette coefficients
k_values = np.linspace(1000, 3600, 11)
k_values = k_values.astype(int)
silhouette_coefficients = []
sse = []
for k in range(len(k_values)):
    kmeans = KMeans(n_clusters=k_values[k], **kmeans_kwargs)
    kmeans.fit(points)
    sse.append(kmeans.inertia_)
    score = silhouette_score(points, kmeans.labels_)
    silhouette_coefficients.append(score)
    print(f'Number of cluster {k+1} out of {len(k_values)} complete')

plt.figure(1)
plt.style.use("fivethirtyeight")
plt.plot(k_values, sse)
plt.xlabel("Number of Clusters")
plt.ylabel("Sum Squared Error")
plt.show()

plt.figure(2)
plt.style.use("fivethirtyeight")
plt.plot(k_values, silhouette_coefficients)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()
