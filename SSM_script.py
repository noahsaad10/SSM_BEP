import matplotlib.pyplot as plt
import numpy as np
from landmark_script import *
from plot_cumulative_variance import *
from plot_shape_modes import *
from stl import mesh
import glob

# This script performs SSM
# Loads the landmarks that are previously generated.
# Author: Noah Saad - n.w.saad@student.tudelft.nl


# Load the generated landmarks to save time
txt_landmark_files = glob.glob('./COW/links/landmarks/*.txt')
#txt_landmark_files = txt_landmark_files[0:2]
print(f'There are {len(txt_landmark_files)} files used in the SSM')
num_landmarks = len(np.loadtxt(txt_landmark_files[0]))

# Build the matrix with all the objects concatenated.
shape_matrix = (num_landmarks, 3)
landmarks_list = np.zeros(((len(txt_landmark_files),) + shape_matrix))
for i in range(len(txt_landmark_files)):
    landmarks = np.loadtxt(txt_landmark_files[i])
    landmarks = landmarks.tolist()
    landmarks_list[i] = landmarks

mean_object = np.mean(landmarks_list, axis=0)

# Run SSM
ssm = pyssam.SSM(landmarks_list)

# Create a PCA model from the SSM landmark
ssm.create_pca_model(ssm.landmarks_columns_scale, desired_variance=0.95)
mean_shape_columnvector = ssm.compute_dataset_mean()
mean_shape = mean_shape_columnvector.reshape(-1, 3)
shape_model_components = ssm.pca_model_components


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D points
ax.scatter(mean_object[:, 0], mean_object[:, 1], mean_object[:, 2], s=2)

# Set axis labels and limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()


print(f"To obtain {ssm.desired_variance*100}% variance, {ssm.required_mode_number} modes are required")
plot_cumulative_variance(np.cumsum(ssm.pca_object.explained_variance_ratio_), 0.95)

# Visualize the three main components

saving_path = r'C:\Users\noah-\Desktop\TU\Year 3\BEP\\'

mode_to_plot = 0
print(f"Explained variance of mode {mode_to_plot} is", round(ssm.pca_object.explained_variance_ratio_[mode_to_plot]*100), "%")
print('---------------------------------------------------------------------')
plot_shape_modes(ssm, mean_shape_columnvector, mean_shape, ssm.model_parameters, ssm.pca_model_components, mode_to_plot, saving_path)




