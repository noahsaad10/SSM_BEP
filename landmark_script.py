import matplotlib.pyplot as plt
import numpy as np
import trimesh.registration
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import open3d as o3d
from stl import mesh
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from trimesh import registration

# The first function generates a reference landmark to later be used for aligning all objects. It uses K-mean clustering
# and generates a landmark for centroid of each cluster.
# Input: STL file and number of landmarks desired
#
# The second function generates landmarks in exactly the same way as the first, and performs Procrustes to align the
# reference with the landmarks. It computes the closest reference points and find the cross covariance matrix between
# the reference and the closest_ref_points to find a transformation that aligns them.
# Uses SVD to find the rotation matrices. Furthermore, it ensures that landmarks are one-to-one correspondent by using
# the Hungarian Algorithm.
# Finally, returns the landmarks.
# Input: STL file to be analyzed, num_landmarks (should be same as reference), and reference landmarks
#
# Author: Noah Saad - n.w.saad@student.tudelft.nl


def reference_landmark(mesh_name, num_landmarks):

    mesh_file = mesh.Mesh.from_file(mesh_name)
    _, com, _ = mesh_file.get_mass_properties()
    mesh_file.translate(-com)
    # Convert the mesh to a numpy array
    points = np.asarray(mesh_file.vectors.reshape(-1,3))

    # Use KMeans to automatically determine landmark positions
    kmeans = KMeans(n_clusters=num_landmarks, random_state=0, n_init=8).fit(points)

    # Get the centroid of each cluster as the landmark position
    landmarks = kmeans.cluster_centers_
    landmarks -= landmarks.mean(axis=0)  # Set the center of landmarks at the origin

    return landmarks


def generate_landmarks(mesh_name, num_landmarks, reference_landmark):

    # Load the STL file using Open3D
    mesh_file = mesh.Mesh.from_file(mesh_name)
    _, com, _ = mesh_file.get_mass_properties()
    mesh_file.translate(-com)
    # Convert the mesh to a numpy array
    points = np.asarray(mesh_file.vectors.reshape(-1, 3))
    # Use KMeans to automatically determine landmark positions
    kmeans = KMeans(n_clusters=num_landmarks, random_state=0, n_init=8).fit(points)
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

    # Create correspondence, ensuring one-to-one mapping
    dist_matrix_correspondence = distance.cdist(reference_landmark, transformed_points)
    row_indices_correspondence, col_indices_correspondence = linear_sum_assignment(dist_matrix_correspondence)
    landmarks_correspondent = transformed_points[col_indices_correspondence]

    return landmarks_correspondent


def generate_landmarks_trimesh(mesh_name, num_landmarks, reference_landmark):

    # Load the STL file using Open3D
    mesh_file = mesh.Mesh.from_file(mesh_name)
    _, com, _ = mesh_file.get_mass_properties()
    mesh_file.translate(-com)
    # Convert the mesh to a numpy array
    points = np.asarray(mesh_file.vectors.reshape(-1, 3))
    # Use KMeans to automatically determine landmark positions
    kmeans = KMeans(n_clusters=num_landmarks, random_state=0, n_init=8).fit(points)
    # Get the centroid of each cluster as the landmark position
    landmarks = kmeans.cluster_centers_

    # Create correspondence, ensuring one-to-one mapping
    dist_matrix_correspondence = distance.cdist(reference_landmark, landmarks)
    row_indices_correspondence, col_indices_correspondence = linear_sum_assignment(dist_matrix_correspondence)
    landmarks_correspondent = landmarks[col_indices_correspondence]

    _, landmarks_transformed, _ = trimesh.registration.procrustes(landmarks_correspondent, reference_landmark, scale=False)

    #dist_matrix_correspondence_2 = distance.cdist(reference_landmark, landmarks_transformed)
    #row_indices_correspondence, col_indices_correspondence = linear_sum_assignment(dist_matrix_correspondence_2)
    #final_landmarks = landmarks_transformed[col_indices_correspondence]

    return landmarks_transformed





