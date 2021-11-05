import numpy as np

__all__ = ['read_obj_file']


def read_obj_file(filename):
    """Reads a Wavefront OBJ file to get the vertices and faces.

    Parameters
    ----------
    filename : `str`
        Path to the OBJ file.

    Returns
    -------
    vertices : `numpy.array`
        2D array (n, 3) with the "n" vertices of the object

    faces : `numpy.array`
        2D array (m, k) with the "k" vertices that makes each of the "m" faces of the object.

    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    vertices = []
    faces = []
    for line in lines:
        if line.startswith('v '):
            vertices.append(line.strip().split(' ')[1:4])
        elif line.startswith('f '):
            faces.append(line.strip().split(' ')[1:])

    vertices = np.array(vertices, dtype='float')
    faces = np.array(faces, dtype='int32')
    return vertices, faces


def read_obj_file2(filename):
    """Reads a Wavefront OBJ file to get the vertices and faces.

    Parameters
    ----------
    filename : `str`
        Path to the OBJ file.

    Returns
    -------
    vertices : `numpy.array`
        2D array (n, 3) with the "n" vertices of the object

    faces : `numpy.array`
        2D array (m, k) with the "k" vertices that makes each of the "m" faces of the object.

    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    vertices = []
    faces = []
    for line in lines:
        if line.startswith('v '):
            vertices.append(line.strip().split(' ')[2:5])
        elif line.startswith('f '):
            faces.append(line.strip().replace('/',' ').split(' ')[2::3])
            #faces.append(line.strip().replace('/',' ').split(' ')[3::3])
            #faces.append(line.strip().replace('/',' ').split(' ')[4::3])

    vertices = np.array(vertices, dtype='float')
    faces = np.array(faces, dtype='int32')
    return vertices, faces

