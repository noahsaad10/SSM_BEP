import matplotlib.pyplot as plt
import numpy as np
from landmark_script import *
from plot_cumulative_variance import *
from plot_shape_modes import *
from vtk_2_stl import *
from stl import mesh
import glob

# This script performs SSM
#
# Author: Noah Saad - n.w.saad@student.tudelft.nl

'''
# Import all VTK meshes
files_vtk = glob.glob('./Right_Hip/groomed/*.vtk')

# Convert all files to STL
for i in range(len(files_vtk)):
    vtk_2_stl(files_vtk[i])
'''

# Import all STL files
files = glob.glob('./Pelvis/*.stl')
files = files  # Eventually if you wish to only select a few files

ref_file = './Pelvis/TrajectoryVolume_10_Superior Ramus Right.stl'
num_landmarks = 100

ref_landmarks = reference_landmark(ref_file, num_landmarks=num_landmarks)

shape_matrix = (num_landmarks, 3)
landmarks_list = np.zeros(((len(files),) + shape_matrix))

for i in range(len(files)):
    landmarks = generate_landmarks(files[i], num_landmarks, reference_landmark= ref_landmarks)
    landmarks = landmarks.tolist()
    landmarks_list[i] = landmarks
    print('File', i+1, 'out of', len(files), 'completed')


'''
# For random points

num_samples = 10
num_landmarks = 10000
# Create random landmarks for SSM
landmarks = np.random.normal(size=(num_samples, num_landmarks, 3))



stl_nodes_num = landmarks_list[0]
vertex_nodes = stl_nodes_num.vectors.reshape(-1, 3)
shape_matrix = (len(vertex_nodes), 3)  # should be constant

# Create the array of matrices for SSM
matrix_data = np.zeros(((len(landmarks_list),) + shape_matrix))

# Go through all files and complete matrix array
for i in range(len(landmarks_list)):
    stl_analyzed = landmarks_list[i]
    stl_data = mesh.Mesh.from_file(stl_analyzed)
    _, com, _ = stl_data.get_mass_properties()
    stl_data.translate(-com)
    vertex_coords = stl_data.vectors.reshape(-1, 3)
    matrix_data[i] = vertex_coords
'''

# Run SSM
ssm = pyssam.SSM(landmarks_list)

# Create a PCA model from the SSM landmark
ssm.create_pca_model(ssm.landmarks_columns_scale, desired_variance=0.95)
mean_shape_columnvector = ssm.compute_dataset_mean()
mean_shape = mean_shape_columnvector.reshape(-1, 3)
shape_model_components = ssm.pca_model_components

print(f"To obtain {ssm.desired_variance*100}% variance, {ssm.required_mode_number} modes are required")
plot_cumulative_variance(np.cumsum(ssm.pca_object.explained_variance_ratio_), 0.95)

# Visualize the three main components

mode_to_plot = 0
print(f"Explained variance of mode {mode_to_plot} is", round(ssm.pca_object.explained_variance_ratio_[mode_to_plot]*100), "%")
print('---------------------------------------------------------------------')
plot_shape_modes(ssm, mean_shape_columnvector, mean_shape, ssm.model_parameters, ssm.pca_model_components, mode_to_plot)




