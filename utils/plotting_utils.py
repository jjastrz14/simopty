'''
==================================================
File: plotting_utils.py
Project: utils
File Created: Thursday, 26th December 2024
Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''

"""

The plotting_utils.py module contains the functions used to plot the results of the simulation.
Those where originally distributed in the different classes of the project.

"""

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from domain import Grid
from mapper import Mapper
from graph import TaskGraph


"""
 = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
 = = = = = = = = = = = = = = = = TASKGRAPH PLOTTING = = = = = = = = = = = = = = = = =
 = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
"""

def plot_graph(graph: TaskGraph, file_path = None):
        """
        Plots the nodes and edges in a top-down fashion.

        Parameters
        ----------
        file_path : str
            The path where to save the plot.
        
        Returns
        -------
        None
        """

        pos = nx.multipartite_layout(graph.graph, subset_key="layer")
        colors = [graph.graph.nodes[node]["color"] for node in graph.graph.nodes()]
        labels = {node: node for node in graph.graph.nodes()}
        node_size = 1000
        node_shape = "s"

        for node_id, node in enumerate(graph.graph.nodes()):
            nx.draw_networkx_nodes(graph.graph, pos, nodelist = [node], node_color = colors[node_id], node_size = node_size, node_shape = node_shape)
            nx.draw_networkx_labels(graph.graph, pos, labels = labels)
        nx.draw_networkx_edges(graph.graph, pos, node_size = node_size, node_shape = node_shape)
        plt.tight_layout()
        plt.axis("off")
        
        if file_path is not None:
            file_path = os.path.join(os.path.dirname(__file__), file_path)
            plt.savefig(file_path)
        plt.show()


"""
 = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
 = = = = = = = = = = = = = = = = GRID PLOTTING = = = = = = = = = = = = = = = = =
 = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
"""

def plot_grid_2D(domain : Grid, file_path = None):
        """
        Plots the grid
        
        Parameters
        ----------
        file_path : str
            The path of the file where the plot is to be saved.

        Returns
        -------
        None
        """
        assert domain.N == 2
        fig, ax = plt.subplots()
        box_size = 0.30
        ax.set_aspect('equal')
        for i in range(domain.size):
            ax.text(domain.grid[i][0], domain.grid[i][1], str(i), ha='center', va='center')
            ax.plot(domain.grid[i][0], domain.grid[i][1], "s", markerfacecolor = 'lightblue', markeredgecolor = 'black', markersize = box_size*100, markeredgebox_size = 2)
        plt.xlim(-box_size-0.5, domain.K-0.5 + box_size)
        plt.ylim(-box_size-0.5, domain.K-0.5 + box_size)
        plt.axis("off")
        
        if file_path is not None:
            file_path = os.path.join(os.path.dirname(__file__), file_path)
            plt.savefig(file_path)
        plt.show()

"""
 = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
 = = = = = = = = = = = = = = = = MAPPING PLOTTING = = = = = = = = = = = = = = = = =
 = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
"""

def plot_mapping_2D(mapper: Mapper, file_path = None):
        """
        Plots the mapping of the tasks onto the NoC grid.

        Parameters
        ----------
        file_path : str
            The path of the file where the plot is to be saved.

        Returns
        -------
        None
        """
        assert mapper.grid.N == 2, "The plot_mapping_2D method is only available for 2D grids."
        fig,ax = plt.subplots()
        box_size = 0.50
        dim_offset = [mapper.grid.K ** i for i in range(mapper.grid.N)]
        # choose a discrete colormap (pastel): a color for each layer
        cmap = plt.cm.get_cmap('Pastel1', len(mapper.dep_graph.nodes.keys()))


        ax.set_aspect('equal')
        for i in range(mapper.grid.K):
            for j in range(mapper.grid.K):
                ax.add_patch(plt.Rectangle((i-box_size/2, j-box_size/2), box_size, box_size, facecolor = 'white', edgecolor = 'black', linewidth = 2, zorder = 0))
                ax.text(i, j - box_size*3/5, f"{i, j}", ha = 'center', va = 'top')
        for i in mapper.dep_graph.nodes.keys():
            mapped_node = mapper.mapping[i]
            layer_id = mapper.dep_graph.nodes[i]["layer_id"]
            color = cmap(layer_id)
            mapped_coords = [mapped_node // dim_offset[h] % mapper.grid.K for h in range(mapper.grid.N)]
            ax.add_patch(plt.Rectangle((mapped_coords[0]-box_size/2, mapped_coords[1]-box_size/2), box_size, box_size, facecolor = color, edgecolor = 'black', linewidth = 2, zorder = 1))
            ax.text(mapped_coords[0], mapped_coords[1], str(i), ha = 'center', va = 'center', color = 'black', fontweight = 'bold')
        plt.xlim(-box_size-0.5, mapper.grid.K-0.5 + box_size)
        plt.ylim(-box_size-0.5, mapper.grid.K-0.5 + box_size)
        plt.axis("off")
        
        if file_path is not None:
            file_path = os.path.join(os.path.dirname(__file__), file_path)
            plt.savefig(file_path)
        plt.show()


def plot_mapping_gif(mapper: Mapper, file_path = None):
    """
    The function is used to create a GIF of the mapping of the tasks onto the NoC grid.
    Each frame in the GIF is used for a different layer (level in the dependency graph): the
    tasks will appear assigned to the PEs of the NoC grid in the order in which they are
    defined in the depency graph (parallel tasks will appear in the same frame).
    """
    assert mapper.grid.N == 2, "The plot_mapping_gif method is only available for 2D grids."
    box_size = 0.50
    dim_offset = [mapper.grid.K ** i for i in range(mapper.grid.N)]
    cmap = plt.cm.get_cmap('Pastel1', len(mapper.dep_graph.nodes.keys()))

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.xlim(-box_size-0.5, mapper.grid.K-0.5 + box_size)
    plt.ylim(-box_size-0.5, mapper.grid.K-0.5 + box_size)
    plt.axis("off")

    layers = set([mapper.dep_graph.nodes[task]["layer_id"] for task in mapper.dep_graph.nodes.keys()])

    patches = []
    texts = []

    for i in range(mapper.grid.K):
            for j in range(mapper.grid.K):
                ax.add_patch(plt.Rectangle((i-box_size/2, j-box_size/2), box_size, box_size, facecolor = 'white', edgecolor = 'black', linewidth = 2, zorder = 0))
                ax.text(i, j - box_size*3/5, f"{i, j}", ha = 'center', va = 'top')

    def init(patches = patches, texts = texts):
        patches = []
        texts = []
        
        return patches + texts

    def update(frame, patches = patches, texts = texts):
        #clear the previous layer boxes
        patches = patches[:mapper.grid.K ** 2]
        texts = texts[:mapper.grid.K ** 2]

        # get the tasks of the current layer
        current_layer = [task for task in mapper.dep_graph.nodes.keys() if mapper.dep_graph.nodes[task]["layer_id"] == frame]
        for i in current_layer:
            mapped_node = mapper.mapping[i]
            color = cmap(frame)
            mapped_coords = [mapped_node // dim_offset[h] % mapper.grid.K for h in range(mapper.grid.N)]
            patches.append(ax.add_patch(plt.Rectangle((mapped_coords[0]-box_size/2, mapped_coords[1]-box_size/2), box_size, box_size, facecolor = color, edgecolor = 'black', linewidth = 2, zorder = 1)))
            texts.append(ax.text(mapped_coords[0], mapped_coords[1], str(i), ha = 'center', va = 'center', color = 'black', fontweight = 'bold'))
        
        return patches + texts
        
    
    ani = animation.FuncAnimation(fig, update, frames = layers, init_func = init, interval = 2000, blit = True, repeat = True)
    if file_path is not None:
        file_path = os.path.join(os.path.dirname(__file__), file_path)
        ani.save(file_path, writer = 'imagemagick', fps = 1)
    plt.show()