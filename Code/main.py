import glob
import open3d as o3d
import numpy as np
from alignment_shapes import align_shapes
from class_SSM import *

# Load the generated landmarks from text files
txt_landmark_files = glob.glob('./COW/boven/landmarks/*.txt')
print(f'There are {len(txt_landmark_files)} files used in the SSM')
num_landmarks = len(np.loadtxt(txt_landmark_files[0]))

# Build the matrix with all the objects concatenated
shape_matrix = (num_landmarks, 3)
landmarks_list = np.zeros(((len(txt_landmark_files),) + shape_matrix))
for i in range(len(txt_landmark_files)):
    landmarks = np.loadtxt(txt_landmark_files[i])
    com = np.mean(landmarks, axis=0)
    landmarks = landmarks - com
    landmarks_list[i] = landmarks

input_shape = landmarks_list[0]

# Align the shapes
aligned_landmarks = align_shapes(landmarks_list)

# Initialize the SSM
ssm = StatisticalShapeModel(aligned_landmarks)

variance_threshold = 0.95  # set the desired threshold

modes_required = ssm.get_modes_for_variance(variance_threshold=variance_threshold)
print(f'There are {modes_required} modes required for 95% variance')
ssm.plot_mode_with_weight(0, 1, 2)  # plot modes with weight


# Create a plot to view the mean shape better in Open3d
mean_shape = ssm.mean_shape
mean_shape = mean_shape.reshape(-1,3)
colors = np.zeros((len(mean_shape), 3))
colors[:len(mean_shape), :] = [0.0, 0.0, 0.0]  # set the color of the reference landmarks to red
# Create Point Cloud
pcd = o3d.geometry.PointCloud()
# Set landmarks in the PCD (concatenate all objects if more than one)
pcd.points = o3d.utility.Vector3dVector(mean_shape)
# Set the color
pcd.colors = o3d.utility.Vector3dVector(colors)
# Print the number of points in the point cloud
print(f"Number of points in point cloud: {len(pcd.points)}")
# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])


# Get the modes
eigenvalues, eigenvectors = ssm.get_modes()
n_modes = None  # select the desired number of modes

# Gives back the reconstructed shape, specify how many modes you want to restrict it to
reconstructed_shape, rmse = ssm.fit_shape(input_shape, n_modes=n_modes)

print(f'RMSE of shape with original: {np.format_float_positional(rmse, precision=3, unique=False, fractional=False)}')

# Plot the reconstructed shape
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reconstructed_shape[:, 0], reconstructed_shape[:, 1], reconstructed_shape[:, 2], color='cyan',
           label=f'Reconstructed Shape with RMSE '
                 f'{np.format_float_positional(rmse, precision=3, unique=False, fractional=False)}')
ax.scatter(input_shape[:,0], input_shape[:,1], input_shape[:,2], color = 'magenta', label='Original Shape')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title(f'Reconstruction of Unseen Geometry without restriction of modes')
plt.legend()
plt.show()

# Obtain the cumulative variance
cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)

print(f'The cumulative variance of mode 1 is {np.round(cumulative_variance[0]*100)}%')

# Determine the number of modes required for the variance threshold
n_modes_threshold = np.argmax(cumulative_variance >= variance_threshold) + 1

# Plot the cumulative variance
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='x')
plt.axhline(y=variance_threshold, color='r', linestyle='--', label=f'{variance_threshold} variance threshold')
plt.xlabel('Number of Modes')
plt.ylabel('Cumulative Variance')
plt.title('Cumulative Variance of Modes')
plt.xlim(0, len(txt_landmark_files))  # there cannot be more modes than files hence limit x-axis to this
plt.legend()
plt.grid(True)
plt.show()

# Calculate p test for all modes on a specific shape

p_values = ssm.calculate_ptest(aligned_landmarks[i])
p_values = p_values[:len(txt_landmark_files)]

plt.hist(p_values, bins=50, color='blue', alpha=0.7)
plt.xlabel('P-values')
plt.ylabel('Frequency')
plt.title('Distribution of P-values')
plt.axvline(x=0.05, color='red', linestyle='--', label='Significance Threshold')
plt.legend()

# Get the bin counts
counts, bins, _ = plt.hist(p_values, bins=50)

# Add the count values on top of each bin
for count, bin_val in zip(counts, bins):
    if count > 0:
        plt.text(bin_val, count, str(int(count)), ha='center', va='bottom')

plt.show()
