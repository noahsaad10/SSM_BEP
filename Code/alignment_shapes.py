import numpy as np
import trimesh.registration
from scipy.spatial import procrustes


def align_shapes(shapes):
    n_shapes = len(shapes)

    # Compute the mean shape as the initial average shape
    average_shape = np.mean(shapes, axis=0)

    # Iterate until convergence
    while True:
        aligned_shapes = []

        # Align each shape with the average shape
        for shape in shapes:
            _, aligned_shape, _ = trimesh.registration.procrustes(shape, average_shape, scale=False)
            aligned_shapes.append(aligned_shape)

        # Update the average shape
        updated_average_shape = np.mean(aligned_shapes, axis=0)

        # Check for convergence
        if np.allclose(average_shape, updated_average_shape):
            break

        average_shape = updated_average_shape

    return np.array(aligned_shapes)

