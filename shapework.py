
# +


shape1= 'Part4.STL'
shape2='Part5.STL'
mesh1 = sw.Mesh(shape1)
mesh2 = sw.Mesh(shape2)
shape2 = sw.Subject()
#mesh2 = sw.Mesh(shape2)

#meshlist = [mesh1, mesh2]
shapelist = [shape1, shape2]
print(shapelist)

# Visualize the mesh 
#sw.plot_meshes(meshlist, use_same_window=False, notebook=False)

# Convert mesh from STL to VTK 

#mesh1_vtk = sw.sw2vtkMesh(mesh1)
#mesh2_vtk = sw.sw2vtkMesh(mesh2)


#mesh1.remesh(numVertices = 7000, adaptivity=1.0)
project = sw.Project()


project.set_subjects(subjects = shapelist)

project.save('test_project.xlsx')


# +
files = glob.glob('./Pelvis/*.STL')


subject_list = []
# Create a subject for each stl file
for i in range(len(files)):
    subject = sw.Subject()
    subject.set_original_filenames([files[i]])
    subject.set_display_name(files[i])
    subject_list.append(subject)




project = sw.Project()
project.set_subjects(subject_list)


project.save('Pelvis_study.xlsx')



# +

sw.plot_meshes(sw.Mesh('./groomed/TrajectoryVolume_10_Superior Ramus Right_groomed.vtk'), use_same_window=False, notebook=False)

sw.Mesh('./groomed/TrajectoryVolume_10_Superior Ramus Right_groomed.vtk').


# +
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy 
from vtk.util.numpy_support import numpy_to_vtk


def vtk_2_stl(file_input):
    # Load the VTK mesh from file
    filename = file_input
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    vtk_mesh = reader.GetOutput()

    # Convert the VTK mesh to a numpy array of vertices and faces
    points = vtk_to_numpy(vtk_mesh.GetPoints().GetData())
    faces = vtk_to_numpy(vtk_mesh.GetPolys().GetData()).reshape(-1, 4)[:, 1:]

    # Create an STL writer
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(filename[0:-4]+'.stl'))

    # Create a polydata object and add the vertices and faces to it
    polydata = vtk.vtkPolyData()
    vertices = vtk.vtkPoints()
    vertices.SetData(numpy_to_vtk(points))
    polydata.SetPoints(vertices)
    triangles = vtk.vtkCellArray()
    for face in faces:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, face[0])
        triangle.GetPointIds().SetId(1, face[1])
        triangle.GetPointIds().SetId(2, face[2])
        triangles.InsertNextCell(triangle)
    polydata.SetPolys(triangles)

    # Write the polydata object to the STL file
    writer.SetInputData(polydata)
    writer.Update()



# +
files = glob.glob('./Right_Hip/groomed/*.vtk')

for i in range(len(files)):
    vtk_2_stl(files[i])

# +
print(sw.Mesh('./Pelvis/groomed/TrajectoryVolume_10_Superior Ramus Right_groomed.vtk').numPoints())
print(sw.Mesh('./Pelvis/groomed/TrajectoryVolume_10_Superior Ramus Right_groomed.stl').numPoints())

print('--------------------------------------------------------------------------------')

print(sw.Mesh('./Pelvis/groomed/TrajectoryVolume_11_Superior Ramus Right_groomed.vtk').numPoints())
print(sw.Mesh('./Pelvis/groomed/TrajectoryVolume_11_Superior Ramus Right_groomed.stl').numPoints())

# +
files = glob.glob('./Right_Hip/groomed/*.stl')

family_1 = []

num_1 = len(sw.Mesh(files[0]).points())
print(num_1)
for i in range(len(files)):
    if sw.Mesh(files[i]).numPoints() == num_1:
        #print('This is good: ', files[i])
        family_1.append(files[i])
    else:
        print('This is shit: ', files[i])
    #print('------------------------------------------------')

print(len(family_1))




# +
files = glob.glob('./Pelvis/groomed/*.stl')

for i in range(len(files)):
    print(files[i])
    print(sw.Mesh(files[i]).centerOfMass())
    print('-------------------------------------------------------')
# -

sw.plot_meshes(sw.Mesh(files[2]), use_same_window=False, notebook=False)

# +
subject = sw.Subject()
subject.set_original_filenames(['./Pelvis/TrajectoryVolume_3_Superior Ramus Right.stl'])
subject.set_landmarks_filenames(['./landmarks.csv'])

project = sw.Project()
project.set_subjects([subject])
project.save('project_test_landmarks.xlsx')

# +
sw.__name__


