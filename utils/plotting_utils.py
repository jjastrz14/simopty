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
import pydot
from utils.partitioner_utils import PartitionInfo


"""
 = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
 = = = = = = = = = = = = = = = = TASKGRAPH PLOTTING = = = = = = = = = = = = = = = = =
 = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
"""

def plot_graph(graph, file_path = None):
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
        else:
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
    box_size = 1.0
    scale = 2
    dim_offset = [mapper.grid.K ** i for i in range(mapper.grid.N)]
    layers = set([mapper.dep_graph.nodes[task]["layer_id"] for task in mapper.dep_graph.nodes.keys()])
    cmap = plt.cm.get_cmap('tab20c', len(layers))

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.xlim(-box_size-0.5, scale*mapper.grid.K-0.5 + box_size)
    plt.ylim(-box_size-0.5, scale*mapper.grid.K-0.5 + box_size)
    plt.axis("off")
    plt.tight_layout()


    global patches, texts
    patches = []
    texts = []

    for i in range(mapper.grid.K):
            for j in range(mapper.grid.K):
                patches.append(plt.Rectangle((scale*i-box_size/2, scale*j-box_size/2), box_size, box_size, facecolor = 'white', edgecolor = 'black', linewidth = 2, zorder = 0))
                texts.append(plt.text(scale*i, scale*j - box_size*3/5, f"{i, j}", ha = 'center', va = 'top', fontdict={'fontsize': 7}))

    def init():
        global patches, texts
        for patch in patches[mapper.grid.K ** 2:]:
            patch.remove()
        for text in texts[mapper.grid.K ** 2:]:
            text.set_text("")
        patches = patches[:mapper.grid.K ** 2]
        texts = texts[:mapper.grid.K ** 2]

        for patch in patches:
            ax.add_patch(patch)
        for text in texts:
            ax.add_artist(text)
        return patches + texts

    def update(frame):
        global patches, texts
        #clear the previous frame
        for patch in patches[mapper.grid.K ** 2:]:
            patch.remove()
        for text in texts[mapper.grid.K ** 2:]:
            text.set_text("")

        patches = patches[:mapper.grid.K ** 2]
        texts = texts[:mapper.grid.K ** 2]

        
        # get the tasks of the current layer
        current_layer = [(task,mapper.dep_graph.nodes[task]["layer_id"]) for task in mapper.dep_graph.nodes.keys() if mapper.dep_graph.nodes[task]["layer_id"] == frame]
        
        for n,(task,layer_id) in enumerate(current_layer):
            mapped_node = mapper.mapping[task]
            color = cmap(frame)
            mapped_coords = [mapped_node // dim_offset[h] % mapper.grid.K for h in range(mapper.grid.N)]
            patches.append(plt.Rectangle((scale*mapped_coords[0]-box_size/2, scale*mapped_coords[1]-box_size/2), box_size, box_size, facecolor = color, edgecolor = 'black', linewidth = 2, zorder = 1))
            ax.add_patch(patches[-1])
            texts.append(plt.text(scale*mapped_coords[0], scale*mapped_coords[1], "L{}-{}".format(layer_id, n), ha = 'center', va = 'center', color = 'black', fontweight = 'bold', fontdict={'fontsize': 6}))
            ax.add_artist(texts[-1])
        
        return patches + texts
        
    
    ani = animation.FuncAnimation(fig, update, frames = layers, init_func = init, interval = 2000, blit = False, repeat = True)
    if file_path is not None:
        file_path = os.path.join(os.path.dirname(__file__), file_path)
        ani.save(file_path, writer = 'magick', fps = 1)
    plt.show()

def plot_partitions(partitions, partitions_deps, namefile = 'visual/task_graph.png'):
    """
    A function to plot the partitions of a layer using pydot package.

    Args:
    - partitions : a dictionary of partitions of the layers

    Returns:
    - a plot representing the task graph that will be deployed on the NoC
    """
    task_id = -1

    def format_node(partition: PartitionInfo):
    
        # we divide the node horizontally in 3 parts: the first part contains the partition id,
        # the second contains the input, output bounds, the number of input and output channels and the weights shape
        # the third part contains the MACs and FLOPs
        struct = f"{partition.id}\ntask_id:{partition.task_id} | layer_type:{type(partition.layer).__name__}\ninput bounds:{partition.in_bounds}\noutput bounds:{partition.out_bounds}\ninput channels:{partition.in_ch}\noutput channels:{partition.out_ch}\nweights shape:{partition.weights_shape} | MACs:{partition.MACs}\nFLOPs:{partition.FLOPs}\ntot_size:{partition.tot_size}"
        if partition.additional_data is not None:
            struct = "{{" + struct
            struct += "}| merged tasks: \n"
            for keys in partition.additional_data.keys():
                struct += f"{keys} \n"
            struct += "}"
        return struct

    # get the type of keras layer


    graph = pydot.Dot(graph_type='digraph')
    for layer, partition_list in partitions.items():
        for partition in partition_list:
            partition.task_id = task_id
            task_id += 1
            if partition.FLOPs > 0:
                node = pydot.Node(partition.id,label = format_node(partition), shape = "Mrecord")
                graph.add_node(node)

    for key, value in partitions_deps.items():
        for dep, weight in value.items():
            if weight > 0:
                edge = pydot.Edge(dep[0], dep[1], label = weight)
                graph.add_edge(edge)

    if namefile:
        _, ext = os.path.splitext(namefile)
        if ext == '.png':
            graph.write_png(namefile)
        elif ext == '.svg':
            graph.write_svg(namefile)
        elif ext == '.pdf':
            graph.write_pdf(namefile)
        else:
            raise TypeError("File format not supported! Check the namefile parameter")
    else:
        raise ValueError("namefile parameter is required in order to save the plot")


