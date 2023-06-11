import numpy as np
import matplotlib.pyplot as plt


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

    def plot_mode_with_weight(self, mode_index, weight):
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
        # Plot the mean shape
        fig = plt.figure()
        ax = fig.add_subplot(1, 3, 2, projection='3d')
        ax.scatter(mean_shape[:, 0], mean_shape[:, 1], mean_shape[:, 2], color='blue', label='Mean Shape')
        ax.set_title('Mean Shape')

        # Plot the modified shape positive
        ax = fig.add_subplot(1, 3, 3, projection='3d')
        ax.scatter(modified_shape_positive[:, 0], modified_shape_positive[:, 1], modified_shape_positive[:, 2], color='black',
                   label='Modified Shape')
        ax.set_title(f'Mode {mode_index + 1} with Weight {weight}')

        # Plot the modified shape negative
        ax = fig.add_subplot(1, 3, 1, projection='3d')
        ax.scatter(modified_shape_negative[:, 0], modified_shape_negative[:, 1], modified_shape_negative[:, 2],
                   color='green',
                   label='Modified Shape')
        ax.set_title(f'Mode {mode_index + 1} with Weight {-1*weight}')

        # Display the plot
        plt.tight_layout()
        plt.show()