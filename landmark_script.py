import numpy as np
from sklearn.cluster import KMeans
import open3d as o3d
from stl import mesh
from scipy.spatial import distance

# The first function generates a reference landmark to later be used for aligning all objects. It uses K-mean clustering
# and generates a landmark for centroid of each cluster.
# Input: STL file and number of landmarks desired
#
# The second function generates landmarks in exactly the same way as the first, and performs Procrustes to align the
# reference with the landmarks. It computes the closest reference points and find the cross covariance matrix between
# the reference and the closest_ref_points to find a transformation that aligns them.
# Uses SVD to find the rotation matrices. Furthermore, it ensures that landmarks are one-to-one correspondent.
# Finally, returns the landmarks.
# Input: STL file to be analyzed, num_landmarks (should be same as reference), and reference landmarks
#
# Note: Bottom lines can be used for visualization purposes.
#
# Author: Noah Saad - n.w.saad@student.tudelft.nl


def reference_landmark(mesh_file, num_landmarks):

    mesh_file = mesh.Mesh.from_file(mesh_file)
    _, com, _ = mesh_file.get_mass_properties()
    mesh_file.translate(-com)
    # Convert the mesh to a numpy array
    points = np.asarray(mesh_file.vectors.reshape(-1,3))

    # Use KMeans to automatically determine landmark positions
    kmeans = KMeans(n_clusters=num_landmarks, random_state=0).fit(points)

    # Get the centroid of each cluster as the landmark position
    landmarks = kmeans.cluster_centers_
    landmarks -= landmarks.mean(axis=0)  # Set the center of landmarks at the origin

    return landmarks


def generate_landmarks(mesh_file, num_landmarks, reference_landmark):

    # Load the STL file using Open3D
    mesh_file = mesh.Mesh.from_file(mesh_file)
    _, com, _ = mesh_file.get_mass_properties()
    mesh_file.translate(-com)
    # Convert the mesh to a numpy array
    points = np.asarray(mesh_file.vectors.reshape(-1, 3))
    # Use KMeans to automatically determine landmark positions
    kmeans = KMeans(n_clusters=num_landmarks, random_state=0).fit(points)
    # Get the centroid of each cluster as the landmark position
    landmarks = kmeans.cluster_centers_

    closest_ref_points = []
    for landmark in landmarks:
        distances = distance.cdist(reference_landmark, [landmark])
        closest_ref_point = reference_landmark[np.argmin(distances)]
        closest_ref_points.append(closest_ref_point)
    closest_ref_points = np.array(closest_ref_points)

    # Perform Procrustes analysis
    mtx1 = closest_ref_points
    mtx2 = landmarks
    # translate to origin
    mtx1 -= mtx1.mean(axis=0)
    mtx2 -= mtx2.mean(axis=0)
    # find optimal rotation
    U, _, Vt = np.linalg.svd(mtx1.T @ mtx2)
    R = Vt.T @ U.T

    # apply rotation, translation, and scaling to landmarks
    transformed_points = (landmarks @ R) + closest_ref_points.mean(axis=0)

    landmarks_correspondent = []
    # Create correspondence, can be improved
    for i in range(len(reference_landmark)):
        distances_correspondent = np.linalg.norm(transformed_points - reference_landmark[i], axis=1)
        correspondent_point = transformed_points[np.argmin(distances_correspondent)]
        landmarks_correspondent.append(correspondent_point)
    landmarks_correspondent = np.array(landmarks_correspondent)

    return landmarks_correspondent


ref_landmarks = reference_landmark('./Geometries/Sacrum/Sacrum_8.stl', 300)

final_landmarks = generate_landmarks('./Geometries/Sacrum/Sacrum_13.stl', 300, ref_landmarks)


colors = np.zeros((len(ref_landmarks) + len(final_landmarks), 3))
colors[:len(ref_landmarks), :] = [1.0, 0.0, 0.0]  # set the color of the reference landmarks to red
colors[len(ref_landmarks):, :] = [0.0, 0.0, 0.0]  # set the color of the transformed landmarks to black

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.concatenate((ref_landmarks, final_landmarks), axis=0))
pcd.colors = o3d.utility.Vector3dVector(colors)

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15, origin=[0, 0, 0])

# Print the number of points in the point cloud
print(f"Number of points in point cloud: {len(pcd.points)/2}")

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd, mesh_frame])
