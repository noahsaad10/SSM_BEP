import glob
from class_SSM import *


# Load the generated landmarks from text files
txt_landmark_files = glob.glob('./Geometries/aorta_landmarks/*.txt')
print(f'There are {len(txt_landmark_files)} files used in the SSM')
num_landmarks = len(np.loadtxt(txt_landmark_files[0]))

# Build the matrix with all the objects concatenated
shape_matrix = (num_landmarks, 3)
landmarks_list = np.zeros(((len(txt_landmark_files),) + shape_matrix))
for i in range(len(txt_landmark_files)):
    landmarks = np.loadtxt(txt_landmark_files[i])
    landmarks_list[i] = landmarks


ssm = StatisticalShapeModel(landmarks_list)

variance_threshold = 0.95
modes_required = ssm.get_modes_for_variance(variance_threshold=variance_threshold)
ssm.plot_mode_with_weight(0, 2)

# Get the first three modes
eigenvalues, eigenvectors = ssm.get_modes()

# Fit a new input shape to the model using two modes
input_shape = landmarks_list[0]

# Gives back the reconstructed shape, specify how many mode you want to restrict it to
reconstructed_shape, rmse = ssm.fit_shape(input_shape, n_modes=None)

print(f'RMSE of shape with original: {np.format_float_positional(rmse, precision=3, unique=False, fractional=False)}')

print(f'There are {modes_required} modes required for 95% variance')


# Plot the reconstructed shape
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reconstructed_shape[:, 0], reconstructed_shape[:, 1], reconstructed_shape[:, 2], color='cyan', label='Reconstructed Shape')
ax.scatter(input_shape[:,0], input_shape[:,1], input_shape[:,2], color = 'magenta', label='Original Shape')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()
plt.show()


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