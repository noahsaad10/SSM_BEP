import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy import stats


class StatisticalShapeModel:
    def __init__(self, training_data):
        self.mean_shape = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.n_modes = None

        # Build the statistical shape model
        self.build_model(training_data)

    def build_model(self, training_data):
        # Flatten the landmark data to (n_samples, n_landmarks * 3)
        flattened_data = training_data.reshape(training_data.shape[0], -1)

        # Compute the mean shape
        self.mean_shape = np.mean(flattened_data, axis=0)

        # Subtract the mean shape to obtain the deviation vectors
        deviations = flattened_data - self.mean_shape

        # Perform Principal Component Analysis (PCA)
        covariance_matrix = np.cov(deviations, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_indices]
        self.eigenvectors = eigenvectors[:, sorted_indices]
        self.n_modes = self.eigenvalues.shape[0]

    def get_modes(self):
        # Return all eigenvalues and eigenvectors
        return self.eigenvalues, self.eigenvectors

    def fit_shape(self, input_shape, n_modes=None):
        # Flatten the input shape to (n_landmarks * 3,)
        flattened_input = input_shape.reshape(-1)

        # Ensure the flattened input shape has the same dimension as the mean shape
        assert flattened_input.shape == self.mean_shape.shape, \
            "Input shape dimension mismatch"

        # Get the desired number of modes
        eigenvalues, eigenvectors = self.get_modes()

        if n_modes is not None:
            eigenvalues = eigenvalues[:n_modes]
            eigenvectors = eigenvectors[:, :n_modes]

        # Subtract the mean shape from the flattened input shape
        deviation = flattened_input - self.mean_shape

        # Compute the shape parameters
        shape_parameters = np.dot(deviation, eigenvectors)

        # Reconstruct the shape using the shape parameters
        reconstructed_shape = self.mean_shape + np.dot(shape_parameters, eigenvectors.T)

        # Reshape the reconstructed shape to (n_landmarks, 3)
        reconstructed_shape = reconstructed_shape.reshape(input_shape.shape)

        rmse = np.sqrt(np.mean((reconstructed_shape - input_shape) ** 2))

        return reconstructed_shape, rmse

    def get_modes_for_variance(self, variance_threshold):
        # Get all eigenvalues
        eigenvalues, _ = self.get_modes()

        # Compute the cumulative variance
        cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)

        # Find the number of modes required to reach the variance threshold
        n_modes = np.argmax(cumulative_variance >= variance_threshold) + 1

        return n_modes

    def plot_mode_with_weight(self, mode_index, weight_1, weight_2):
        # Get the mean shape and eigenvectors
        mean_shape = self.mean_shape
        eigenvectors = self.eigenvectors

        # Reshape the eigenvectors to match the shape of the mean shape
        eigenvectors_reshaped = eigenvectors.reshape(mean_shape.shape[0], -1)

        # Compute the shape variation using the mode and weight
        std = np.sqrt(self.eigenvalues[mode_index]) # Standard deviation
        shape_variation_positive_1 = std *weight_1 * eigenvectors_reshaped[:, mode_index]
        shape_variation_negative_1 = -1*weight_1 * std * eigenvectors_reshaped[:, mode_index]

        #shape_variation_positive_2 = std * weight_2 * eigenvectors_reshaped[:, mode_index]
        #shape_variation_negative_2 = -1*weight_2 * std * eigenvectors_reshaped[:, mode_index]
        # Combine the mean shape and shape variation
        modified_shape_positive_1 = mean_shape + shape_variation_positive_1
        modified_shape_negative_1 = mean_shape + shape_variation_negative_1

        #modified_shape_positive_2 = mean_shape + shape_variation_positive_2
        #modified_shape_negative_2 = mean_shape + shape_variation_negative_2

        # Reshape the mean shape to (n_landmarks,3)
        mean_shape = mean_shape.reshape(-1,3)
        # Reshape the modified shape to (n_landmarks, 3)
        modified_shape_positive_1 = modified_shape_positive_1.reshape(-1,3)
        modified_shape_negative_1 = modified_shape_negative_1.reshape(-1, 3)
        #modified_shape_positive_2 = modified_shape_positive_2.reshape(-1,3)
        #modified_shape_negative_2 = modified_shape_negative_2.reshape(-1, 3)
        # Plot the mean shape
        fig = plt.figure()
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        ax.scatter(mean_shape[:, 0], mean_shape[:, 1], mean_shape[:, 2], color='gray', label='Mean Shape', s=1)
        ax.set_title('Mean Shape')

        # Plot the modified shape positive
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        ax.scatter(modified_shape_positive_1[:, 0], modified_shape_positive_1[:, 1], modified_shape_positive_1[:, 2], color='blue',
                   label='Modified Shape', s=1)
        ax.set_title(f'Mode {mode_index + 1} with Weight {weight_1}')

        # Plot the modified shape negative
        ax = fig.add_subplot(2, 2, 3, projection='3d')
        ax.scatter(modified_shape_negative_1[:, 0], modified_shape_negative_1[:, 1], modified_shape_negative_1[:, 2],
                   color='green',
                   label='Modified Shape', s=1)
        ax.set_title(f'Mode {mode_index + 1} with Weight {-1*weight_1}')

        # Plot shape positive weight 2
        #ax = fig.add_subplot(1, 5, 5, projection='3d')
        #ax.scatter(modified_shape_positive_2[:, 0], modified_shape_positive_2[:, 1], modified_shape_positive_2[:, 2], color='blue',
        #           label='Modified Shape', s=1)
        #ax.set_title(f'Mode {mode_index + 1} with Weight {weight_2}')

        # Plot the modified shape negative
        #ax = fig.add_subplot(1, 5, 1, projection='3d')
        #ax.scatter(modified_shape_negative_2[:, 0], modified_shape_negative_2[:, 1], modified_shape_negative_2[:, 2],
        #          color='green',
        #           label='Modified Shape', s=1)
        #ax.set_title(f'Mode {mode_index + 1} with Weight {-1*weight_2}')

        # Display the plot
        plt.tight_layout()
        plt.show()


    def plot_mode_with_weight_open3d(self, mode_index, weight):
        # Get the mean shape and eigenvectors
        mean_shape = self.mean_shape
        eigenvectors = self.eigenvectors

        # Reshape the eigenvectors to match the shape of the mean shape
        eigenvectors_reshaped = eigenvectors.reshape(mean_shape.shape[0], -1)

        # Compute the shape variation using the mode and weight
        std = np.sqrt(self.eigenvalues[mode_index]) # Standard deviation
        shape_variation_positive = std *weight * eigenvectors_reshaped[:, mode_index]
        shape_variation_negative = -1*weight * std * eigenvectors_reshaped[:, mode_index]
        # Combine the mean shape and shape variation
        modified_shape_positive = mean_shape + shape_variation_positive

        modified_shape_negative = mean_shape + shape_variation_negative

        # Reshape the mean shape to (n_landmarks,3)
        mean_shape = mean_shape.reshape(-1,3)
        # Reshape the modified shape to (n_landmarks, 3)
        modified_shape_positive = modified_shape_positive.reshape(-1,3)
        modified_shape_negative = modified_shape_negative.reshape(-1, 3)

        modified_shape_positive[:, 0] += 100
        modified_shape_negative[:, 0] -= 100

        colors = np.zeros((3*len(mean_shape), 3))
        colors[:len(mean_shape), :] = [1.0, 0.0, 0.0]  # set the color of the reference landmarks to red
        colors[len(mean_shape):2 * len(mean_shape), :] = [0.0, 0.0, 0.0]
        colors[2 * len(mean_shape):3 * len(mean_shape), :] = [0.0, 1.0, 0.0]
        pcd = o3d.geometry.PointCloud()
        # Set landmarks in the PCD (concatenate all objects if more than one)
        pcd.points = o3d.utility.Vector3dVector(
            np.concatenate([mean_shape, modified_shape_positive, modified_shape_negative], axis=0))
        # Set the color
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])


    def calculate_ptest(self, input_shape):
        # Flatten the input shape to (n_landmarks * 3,)
        flattened_input = input_shape.reshape(-1)

        # Ensure the flattened input shape has the same dimension as the mean shape
        assert flattened_input.shape == self.mean_shape.shape, \
            "Input shape dimension mismatch"

        # Subtract the mean shape from the flattened input shape
        deviation = flattened_input - self.mean_shape

        # Compute the shape parameters
        shape_parameters = np.dot(deviation, self.eigenvectors)

        # Compute the p-values for each shape parameter
        p_values = 1.0 - np.abs(stats.norm.cdf(shape_parameters))

        return p_values

