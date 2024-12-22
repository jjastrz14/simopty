'''
==================================================
File: mapper.py
Project: simopty
File Created: Monday, 9th December 2024
Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''

"""
The mapper.py module contains the classes used to perform the mapping of a split-compiled neural gridwork onto a k-dimensional NoC grid.
The mapper takes as input a list of elements, representing to which PE of the NoC a certain task of the dependecy graph gets mapped to.
It then produces a JSON-like list of elements, representing the mapping of the tasks onto the NoC and the communication between them.
It also produces a graphical representation of the mapping.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Mapper:

    def init(self, dep_graph, grid):
        """
        Initializes the Mapper object with the given parameters.

        Parameters
        ----------
        dep_graph : DepGraph
            The dependency graph to be mapped.
        grid : Grid
            The NoC grid on which the mapping is to be performed.

        Returns
        -------
        None
        """
        self.dep_graph = dep_graph
        self.grid = grid
        self.mapping = None

    def set_mapping(self, mapping):
        """
        Sets the mapping of the tasks onto the NoC grid.

        Parameters
        ----------
        mapping : dict
            A dictionary representing the mapping of the tasks onto the NoC grid:
            mapping = {job_id : node_id}

        Example
        -------
        mapping = {0 : 0, 1 : 3, 2 : 4, ...}

        Returns
        -------
        None
        """
        self.mapping = mapping


    def mapping_to_json(self, file_path, file_to_append = None):
        """
        Produces a JSON-like list of elements representing the mapped dependency graph on the NoC grid
        and writes it to a file.

        Parameters
        ----------
        file_path : str
            The path of the file where the JSON-like list is to be saved.

        Returns
        -------
        None
        """

        # Clear the previous mapping and set the new one
        self.dep_graph.clear_mapping()
        self.dep_graph.apply_mapping(self.mapping)
        # Write the mapping to a file, using the JSON format
        structure = self.dep_graph.graph_to_json()
        path = os.path.join(os.path.dirname(__file__), file_path)
        # write the JSON-like list to a file
        if not file_to_append:
            with open(path, "w") as f:
                json.dump(structure, f, indent = 4)
        else:
            with open(file_to_append, "r") as f:
                data = json.load(f)

            data["arch"]["n"] = self.grid.N
            data["arch"]["k"] = self.grid.K
            data["arch"]["topology"] = self.grid.topology.value
            
            data["workload"] = structure
            with open(path, "w") as f:
                json.dump(data, f, indent = 4)


    def plot_mapping_2D(self, file_path = None):
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
        assert self.grid.N == 2, "The plot_mapping_2D method is only available for 2D grids."
        fig,ax = plt.subplots()
        box_size = 0.50
        dim_offset = [self.grid.K ** i for i in range(self.grid.N)]
        # choose a discrete colormap (pastel): a color for each layer
        cmap = plt.cm.get_cmap('Pastel1', len(self.dep_graph.nodes.keys()))


        ax.set_aspect('equal')
        for i in range(self.grid.K):
            for j in range(self.grid.K):
                ax.add_patch(plt.Rectangle((i-box_size/2, j-box_size/2), box_size, box_size, facecolor = 'white', edgecolor = 'black', linewidth = 2, zorder = 0))
                ax.text(i, j - box_size*3/5, f"{i, j}", ha = 'center', va = 'top')
        for i in self.dep_graph.nodes.keys():
            mapped_node = self.mapping[i]
            layer_id = self.dep_graph.nodes[i]["layer_id"]
            color = cmap(layer_id)
            mapped_coords = [mapped_node // dim_offset[h] % self.grid.K for h in range(self.grid.N)]
            ax.add_patch(plt.Rectangle((mapped_coords[0]-box_size/2, mapped_coords[1]-box_size/2), box_size, box_size, facecolor = color, edgecolor = 'black', linewidth = 2, zorder = 1))
            ax.text(mapped_coords[0], mapped_coords[1], str(i), ha = 'center', va = 'center', color = 'black', fontweight = 'bold')
        plt.xlim(-box_size-0.5, self.grid.K-0.5 + box_size)
        plt.ylim(-box_size-0.5, self.grid.K-0.5 + box_size)
        plt.axis("off")
        
        if file_path is not None:
            file_path = os.path.join(os.path.dirname(__file__), file_path)
            plt.savefig(file_path)
        plt.show()


    def plot_mapping_gif(self, file_path = None):
        """
        The function is used to create a GIF of the mapping of the tasks onto the NoC grid.
        Each frame in the GIF is used for a different layer (level in the dependency graph): the
        tasks will appear assigned to the PEs of the NoC grid in the order in which they are
        defined in the depency graph (parallel tasks will appear in the same frame).
        """
        assert self.grid.N == 2, "The plot_mapping_gif method is only available for 2D grids."
        box_size = 0.50
        dim_offset = [self.grid.K ** i for i in range(self.grid.N)]
        cmap = plt.cm.get_cmap('Pastel1', len(self.dep_graph.nodes.keys()))

        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        plt.xlim(-box_size-0.5, self.grid.K-0.5 + box_size)
        plt.ylim(-box_size-0.5, self.grid.K-0.5 + box_size)
        plt.axis("off")

        layers = set([self.dep_graph.nodes[task]["layer_id"] for task in self.dep_graph.nodes.keys()])

        patches = []
        texts = []

        for i in range(self.grid.K):
                for j in range(self.grid.K):
                    ax.add_patch(plt.Rectangle((i-box_size/2, j-box_size/2), box_size, box_size, facecolor = 'white', edgecolor = 'black', linewidth = 2, zorder = 0))
                    ax.text(i, j - box_size*3/5, f"{i, j}", ha = 'center', va = 'top')

        def init(patches = patches, texts = texts):
            patches = []
            texts = []
            
            return patches + texts

        def update(frame, patches = patches, texts = texts):
            #clear the previous layer boxes
            patches = patches[:self.grid.K ** 2]
            texts = texts[:self.grid.K ** 2]

            # get the tasks of the current layer
            current_layer = [task for task in self.dep_graph.nodes.keys() if self.dep_graph.nodes[task]["layer_id"] == frame]
            for i in current_layer:
                mapped_node = self.mapping[i]
                color = cmap(frame)
                mapped_coords = [mapped_node // dim_offset[h] % self.grid.K for h in range(self.grid.N)]
                patches.append(ax.add_patch(plt.Rectangle((mapped_coords[0]-box_size/2, mapped_coords[1]-box_size/2), box_size, box_size, facecolor = color, edgecolor = 'black', linewidth = 2, zorder = 1)))
                texts.append(ax.text(mapped_coords[0], mapped_coords[1], str(i), ha = 'center', va = 'center', color = 'black', fontweight = 'bold'))
            
            return patches + texts
            
        
        ani = animation.FuncAnimation(fig, update, frames = layers, init_func = init, interval = 2000, blit = True, repeat = True)
        if file_path is not None:
            file_path = os.path.join(os.path.dirname(__file__), file_path)
            ani.save(file_path, writer = 'imagemagick', fps = 1)
        plt.show()

        




        

        



