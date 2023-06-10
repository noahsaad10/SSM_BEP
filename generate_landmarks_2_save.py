import numpy as np

from landmark_script import *
import glob

# This script uses the landmarks script and generates landmarks to save. Specify a saving path.
# Author - Noah Saad (n.w.saad@student.tudelft.nl)

files = glob.glob('./COW/links/groomed/*.stl')
files = files

num_landmarks = 3500
saving_path = r"C:\Users\noah-\Desktop\TU\Year 3\BEP\COW\links\landmarks\\"

ref_landmarks = reference_landmark('./COW/links/groomed/293-Links_groomed.stl', num_landmarks)

for i in range(len(files)):
    file_name = files[i][20:-4]  # for complete COW [14:-4]
    landmarks = generate_landmarks_trimesh(files[i], num_landmarks, ref_landmarks)
    np.savetxt(f"{saving_path}landmarks_{file_name}.txt", landmarks)
    print(f"File {i + 1} out of {len(files)} completed")



'''
for i in range(len(stl_files)):
    file_name = stl_files[i][6:-4]  # Only keeps the patient number
    landmarks = generate_landmarks(stl_files[i], num_landmarks, reference_landmark=ref_landmarks)
    np.savetxt(f"{saving_path}landmarks_{file_name}.txt", landmarks)
    print(f"File {i + 1} out of {len(stl_files)} completed")
'''

