import vtk
import glob
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk


# Function converts vtk files to stl
# Author - Noah Saad (n.w.saad@student.tudelft.nl)


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
    writer.SetFileName(str(filename[0:-4] + '.stl'))

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


files = glob.glob('./COW/links/groomed/*.vtk')

for i in range(len(files)):
    vtk_2_stl(files[i])

