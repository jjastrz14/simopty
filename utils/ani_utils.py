'''
==================================================
File: ani_utils.py
Project: utils
File Created: Thursday, 23rd January 2025
Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''


"""
The ani_utils.py module contains the functions used to generate the animation of the activity on the NoC grid.
The general framework to create the animation as been taken from https://github.com/jmjos/ratatoskr/blob/master/bin/plot_network.py
"""

import matplotlib.pyplot
from typing import Union

# This script generates simple topology files for 2D or 3D meshes
###############################################################################
import sys
import os
import numpy as np
import json
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
###############################################################################
# Global variables
fig = None  # Figure Object
ax = None  # Axes Object
topology = None  # The topology of the mesh
points = {}  # List of the nodes/points
excluded_points = []  # Those are processing elemnts
connections = []  # List of the connection between the points
num_of_layers = 0  # The number of layers in the mesh
# Note: a face is not a layer, but the number of faces equals the number of layers.
# A face consists of the corner points of the layer only.
# That means each face consists of only four points.
layers = []  # list of the layers
faces = []  # List of the faces, for drawing reasons
###############################################################################


def init(config_file):
    """
    Initialize the script by reading the mesh information from the mesh xml file
    """

    def get_neighbors(k,n,i, topology):
        neighbors = []
        if topology == "mesh":
            if i % k != 0:
                neighbors.append((i-1, 0))
            if i % k != k-1:
                neighbors.append((i+1, 0))
            if i // k != 0:
                neighbors.append((i-k, 0))
            if i // k != k-1:
                neighbors.append((i+k, 0))
            return neighbors
        elif topology == "torus":
            if i % k != 0:
                neighbors.append((i-1, 0))
            else:
                neighbors.append((k-1, 1)) # 1 is for left wrap-around
            if i % k != k-1:
                neighbors.append((i+1, 0))
            else:
                neighbors.append((0, 2)) # 2 is for right wrap-around
            if i // k != 0:
                neighbors.append((i-k, 0))
            else:
                neighbors.append((k*(k-1)+i, 3)) # 3 is for top wrap-around
            if i // k != k-1:
                neighbors.append((i+k, 0))
            else:
                neighbors.append((i % k, 4)) # 4 is for bottom wrap-around
            
        else:
            raise RuntimeError('topology type is not supported')
        return neighbors
    
    # read the arch field from configuration file
    config = json.load(open(config_file))
    arch = config['arch']

    # Number of layers
    global num_of_layers
    num_of_layers = 2 # one layer for NoC elements, the other for NPUs
    proc_elemnt_ids = []
    for pe in range(arch["k"]** arch["n"]):
        proc_elemnt_ids.append(pe)


    # Points is a list of tuples
    global points
    points[0] = []  # NoC elements
    points[1] = []  # NPUs
    for p in range(arch["k"]** arch["n"]):
        x = p % arch["k"]
        y = p // arch["k"]
        z = 0
        points[0].append((x, y, z)) # 0 is for NoC elements
        points[1].append((x, y, z+1)) # 1 is for NPUs

    # Build the list of connections based on the mesh topology
    global connections, topology
    topology = arch["topology"]
    for node in range(arch["k"]** arch["n"]):
        neighbors = get_neighbors(arch["k"], arch["n"], node, topology)
        for neighbor in neighbors:
            connections.append([(node, neighbor[0]), neighbor[1]])
    

    
###############################################################################


def create_fig():
    """
    Create the figure object
    """
    global fig
    fig = plt.figure()
    global ax
    ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
###############################################################################

def vertial_connection(p1_ix, p2_ix):
    """
    Draws the vertical connection of the points
    """
    x = []
    y = []
    z = []

    x.append(points[0][p1_ix][0])
    x.append(points[1][p2_ix][0])

    y.append(points[0][p1_ix][1])
    y.append(points[1][p2_ix][1])

    z.append(points[0][p1_ix][2])
    z.append(points[1][p2_ix][2])

    ax.plot(x, y, z, marker='o', color='black')


def horizontal_connection(p1_ix, p2_ix, in_plane):
    """
    Draws the horizontal connection of the points
    """
    x = []
    y = []
    z = []
 
    if in_plane == 0:
        x.append(points[0][p1_ix][0])
        x.append(points[0][p2_ix][0])

        y.append(points[0][p1_ix][1])
        y.append(points[0][p2_ix][1])

        z.append(points[0][p1_ix][2])
        z.append(points[0][p2_ix][2])

        ax.plot(x, y, z, marker='o', color='black')

    else:
        # if the in_plane flag is 0 (this can only happen in the torus topology)
        # this means that the considered connection is a wrap-around connection:
        # we represent this by drawing a line from the two points to the border of the mesh

        border = 0.5

        segment1 = [(points[0][p1_ix][0], points[0][p1_ix][1], points[0][p1_ix][2]), (points[0][p1_ix][0] + (- border if in_plane == 1 else border if in_plane == 2 else 0), points[0][p1_ix][1] + (border if in_plane == 3 else - border if in_plane == 4 else 0), points[0][p1_ix][2])]
        segment2 = [(points[0][p2_ix][0], points[0][p2_ix][1], points[0][p2_ix][2]), (points[0][p2_ix][0] + ( border if in_plane == 1 else - border if in_plane == 2 else 0), points[0][p2_ix][1] + (- border if in_plane == 3 else border if in_plane == 4 else 0), points[0][p2_ix][2])]

        ax.plot([segment1[0][0], segment1[1][0]], [segment1[0][1], segment1[1][1]], [segment1[0][2], segment1[1][2]], marker='o', color='black')
        ax.plot([segment2[0][0], segment2[1][0]], [segment2[0][1], segment2[1][1]], [segment2[0][2], segment2[1][2]], marker='o', color='black')

###############################################################################



###############################################################################


def plot_connections():
    """
    Plot the connections between the nodes/points
    """
    for p in range(len(points[0])):
        vertial_connection(p, p)

    for c in connections:
        p1_ix, p2_ix = c[0]
        in_plane= c[1]

        horizontal_connection(p1_ix, p2_ix, in_plane)
###############################################################################


def annotate_points():
    """
    Annotating the points using their index
    """
    points_coordinates = np.array(points[1])
    i = 0
    for x, y, z in zip(points_coordinates[:, 0], points_coordinates[:, 1], points_coordinates[:, 2]):
        ax.text(x, y, z, i, size=12, color='red')
        i = i + 1
###############################################################################


def create_faces():
    """
    Create the faces of the mesh, each layer will become a face
    """
    # Make layers
    global layers
    global topology

    # Separate lists of x, y and z coordinates
    x_s = []
    y_s = []
    z_s = []

    for i in range(0, num_of_layers):
        layer = []

        for p in points[i]:
                layer.append(p)
                x_s.append(p[0])
                y_s.append(p[1])
                if (p[2] not in z_s):
                    z_s.append(p[2])

        layers.append(layer)

    # Making faces, only out of the corner points of the layer
    global faces
    if topology == "mesh":
        for i in range(0, num_of_layers):
            x_min = min(x_s)
            x_max = max(x_s)
            y_min = min(y_s)
            y_max = max(y_s)
            face = list([(x_min, y_min, z_s[i]), (x_max, y_min, z_s[i]), (x_max, y_max, z_s[i]), (x_min, y_max, z_s[i])])
            faces.append(face)
    elif topology == "torus":
        added_border = 0.5
        for i in range(0, num_of_layers):
            x_min = min(x_s) - added_border
            x_max = max(x_s) + added_border
            y_min = min(y_s) - added_border
            y_max = max(y_s) + added_border
            face = list([(x_min, y_min, z_s[i]), (x_max, y_min, z_s[i]), (x_max, y_max, z_s[i]), (x_min, y_max, z_s[i])])
            faces.append(face)
###############################################################################


def plot_faces():
    """
    Plot the faces
    """
    # Create multiple 3D polygons out of the faces
    poly = Poly3DCollection(faces, linewidths=1, alpha=0.1)
    faces_colors = []
    # Generating random color for each face
    for i in range(0, num_of_layers):
        red = np.random.randint(0, 256)
        green = np.random.randint(0, 256)
        blue = np.random.randint(0, 256)
        color = (red, green, blue)
        if color not in faces_colors:
            faces_colors.append('#%02x%02x%02x' % color)
    poly.set_facecolors(faces_colors)
    ax.add_collection3d(poly)
###############################################################################


def main():
    """
    Main Execution Point
    """
    curdir = os.path.dirname(__file__)
    network_file = curdir +'/../config_files/arch.json'

    try:
        network_file = sys.argv[1]
    except IndexError:
        pass
    init(network_file)
    create_fig()
    plot_connections()
    annotate_points()
    create_faces()
    plot_faces()
    plt.show()
###############################################################################


def gen_activity_animation(logger, file_name: Union[str, None] = None):
    """
    The function takes as input a logger object (defined in the nocsim module, exposed in restart). This is used as a chonological timeline
    on which the events that happen on NoC are registered. We unroll this timeline and plot it in a matplotlib animation

    Args:
     - logger: nocsim.EventLogger object
     - fine_name: a string, 

    Returns:
     - None
    """
    pass


main()

