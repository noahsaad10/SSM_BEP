import numpy as np
from landmark_script import *
import glob

# Visualize landmarks in 3D. Reference landmark in red and transformed landmarks in black.
# Author - Noah Saad (n.w.saad@student.tudelft.nl)

#ref_landmarks = reference_landmark('./COW/links/groomed/123-Links_groomed.stl', 3000)
#files = glob.glob('./COW/landmarks/*.txt')
ref_landmarks = np.loadtxt('./COW/links/landmarks/landmarks_155-Links_groomed.txt')
#final_landmarks = np.loadtxt(files[2])

#final_landmarks = generate_landmarks_trimesh('./COW/links/groomed/155-Links_groomed.stl', 3000, ref_landmarks)
final_landmarks = np.loadtxt('./COW/links/landmarks/landmarks_1588-links_groomed.txt')
#final_landmarks = generate_landmarks_trimesh(r'C:\Users\noah-\Desktop\TU\Year 3\BEP\COW\groomed\193_groomed.stl', 3000, ref_landmarks)


# Select the colors for each object
colors = np.zeros((len(ref_landmarks)+len(final_landmarks), 3))
colors[:len(ref_landmarks), :] = [1.0, 0.0, 0.0]  # set the color of the reference landmarks to red
colors[len(ref_landmarks):, :] = [0.0, 0.0, 0.0]


# Create Point Cloud
pcd = o3d.geometry.PointCloud()
# Set landmarks in the PCD (concatenate all objects if more than one)
pcd.points = o3d.utility.Vector3dVector(np.concatenate([ref_landmarks, final_landmarks], axis=0))
# Set the color
pcd.colors = o3d.utility.Vector3dVector(colors)
# Display the origin
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15, origin=[0, 0, 0])

# Print the number of points in the point cloud
print(f"Number of points in point cloud: {len(pcd.points)/2}")

print('Number of points final: ', len(final_landmarks))

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd, mesh_frame])
