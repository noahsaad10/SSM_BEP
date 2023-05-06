import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pyssam
from copy import copy
import open3d as o3d
import numpy as np
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def plot_shape_modes(ssm,
  mean_shape_columnvector,
  mean_shape,
  original_shape_parameter_vector,
  shape_model_components, mode_to_plot
):
  weights = [-1, 0, 1]
  fig= plt.figure()
  for j, weights_i in enumerate(weights):
    shape_parameter_vector = copy(original_shape_parameter_vector)
    shape_parameter_vector[mode_to_plot] = weights_i
    mode_i_coords = ssm.morph_model(
        mean_shape_columnvector,
        shape_model_components,
        shape_parameter_vector
    ).reshape(-1, 3)

    offset_dist = pyssam.utils.euclidean_distance(
      mean_shape,
      mode_i_coords
    )
    # colour points blue if closer to point cloud centre than mean shape
    mean_shape_dist_from_centre = pyssam.utils.euclidean_distance(
      mean_shape,
      np.zeros(3),
    )
    mode_i_dist_from_centre = pyssam.utils.euclidean_distance(
      mode_i_coords,
      np.zeros(3),
    )
    offset_dist = np.where(
        mode_i_dist_from_centre<mean_shape_dist_from_centre,
        offset_dist*-1,
        offset_dist,
    )

    if weights_i == 0:
      ax = fig.add_subplot(1, 3, 2, projection='3d')
      ax.scatter(
        mode_i_coords[:, 0],
        mode_i_coords[:, 1],
        mode_i_coords[:, 2],
        c="gray",
        s=1,
      )
      ax.set_title("mean shape")
      ax.axis()

    elif weights_i == weights[0]:
      ax = fig.add_subplot(1, 3, 1, projection='3d')
      ax.scatter(
        mode_i_coords[:, 0],
        mode_i_coords[:, 1],
        mode_i_coords[:, 2],
        c=offset_dist,
        cmap="seismic",
        vmin=-1,
        vmax=1,
        s=1,
      )
      ax.set_title(f"mode {mode_to_plot} \nweight {weights_i}")
      ax.axis()

    elif weights_i == weights[2]:
      ax = fig.add_subplot(1, 3, 3, projection='3d')

      ax.scatter(
        mode_i_coords[:, 0],
        mode_i_coords[:, 1],
        mode_i_coords[:, 2],
        c=offset_dist,
        cmap="seismic",
        vmin=-1,
        vmax=1,
        s=1,)

      ax.set_title(f"mode {mode_to_plot} \nweight {weights_i}")
      ax.axis()

  plt.show()

