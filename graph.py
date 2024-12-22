'''
==================================================
File: graph.py
Project: simopty
File Created: Sunday, 8th December 2024
Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''

"""
The graph.py module defines the class DepGraph, which is used to represent a dependency graph.
Specifically, we use it as a wrapper for the networkx.DiGraph class, to represent a directed graph. 
The DepGraph class can be initialized direcly from a JSON-like list of elements, or progressively by adding tasks and dependencies.
representing the computation and communication operations of a split-compiled neural network.
"""

import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class DepGraph:

    NODE_INFO = "Node ID: {0}\nType: {1}\nLayer ID: {2}\nComputing Time Required: {3}\nDependencies: {4}"
    EDGE_INFO = "Edge ID: {0}\nType: {1}\nData Size: {2}\nProcessing Time Required: {3}\nDependencies: {4}"


    def __init__(self):
        self.graph = nx.DiGraph()
        self.SOURCE_POINT = 0
        self.DRAIN_POINT = 8
        # Starting and ending nodes to group starting and ending dependencies
        self.graph.add_node("start", node = self.SOURCE_POINT, type = "START", layer = -1, color = 'lightgreen')
        self.graph.add_node("end", node = self.DRAIN_POINT, type = "END", layer = 999999, color = 'indianred')
        self.nodes = {}
        self.nodes_mapping = {}
        self.edges = {}
        self.edges_mapping = {}
        self.id = 0

    @property
    def n_nodes(self):
        """
        Returns the number of nodes (tasks) in the graph.
        """
        return len(self.nodes)
    
    @property
    def n_edges(self):
        """
        Returns the number of edges (dependencies) in the graph.
        """
        return len(self.edges)

    def clear(self):
        """
        Clears the graph.

        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """

        self.graph.clear()
        self.graph.add_node("start", node = 0, type = "START", layer = -1, color = 'lightgreen')
        self.graph.add_node("end", node = 8, type = "END", layer = np.inf, color = 'indianred')
        self.nodes = {}
        self.nodes_mapping = {}
        self.edges = {}
        self.edges_mapping = {}
        self.id = 0

    def clear_mapping(self):
        """
        Clears the mapping of the tasks onto the NoC grid.

        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """

        self.nodes_mapping = {}
        self.edges_mapping = {}

    def apply_mapping(self, mapping):
        """
        Sets the mapping of the tasks onto the NoC grid.

        Parameters
        ----------
        mapping : dict
            A dictionary representing the mapping of the tasks onto the NoC grid:
            mapping = {job_id : node_id}
        
        Returns
        -------
        self.nodes_mapping : dict
            A dictionary representing the mapping of the nodes onto the NoC grid:
            nodes_mapping = {node_id : node}
        self.edges_mapping : dict
            A dictionary representing the mapping of the edges onto the NoC grid:
            edges_mapping = {edge_id : {"src": src, "dst": dst}}
        """

        self.nodes_mapping = mapping
        for edge_id in self.edges.keys():
            self.edges_mapping[edge_id] = {"src": None, "dst": None}
        # Check that all the nodes are in the mapping
        assert set(self.nodes.keys()).issubset(set(self.nodes_mapping.keys())), "Some nodes are missing in the mapping."
        
        for node_id, node in self.nodes.items():
            # if any of the node's dependencies are in the edges list, set the "dst" attribute of the edge
            # to the node's "node" attribute
            for dep in node["dep"]:
                if dep in self.edges.keys():
                    self.edges_mapping[dep]["dst"] = self.nodes_mapping[node_id]
        for edge_id, edge in self.edges.items():
            # if any of the edge's dependencies are in the nodes list, set the "src" attribute of the edge
            # to the node's "node" attribute
            for dep in edge["dep"]:
                if dep == -1:
                    edge["src"] = self.SOURCE_POINT
                elif dep in self.nodes.keys():
                    self.edges_mapping[edge_id]["src"] = self.nodes_mapping[dep]

        # If, at the end of the mapping, there are still some edges without a "src" or "dst" attribute,
        # set them to "start" and "end" respectively
        for edge in self.edges_mapping.values():
            if edge["src"] is None:
                edge["src"] = self.SOURCE_POINT
            if edge["dst"] is None:
                edge["dst"] = self.DRAIN_POINT

        return self.nodes_mapping, self.edges_mapping


    def add_task(self, id, layer_id, color = "lightblue"):
        """
        Adds a task (node) to the graph.

        Parameters
        ----------
        id : int
            The id of the task. If None, the id is automatically assigned.
        type : str 
            The type of the task.
        layer_id : int
            The layer id of the task.
        color : str
            The color of the task.
        """
        if id is None:
            id = self.id
            self.id += 1
        self.graph.add_node(id, layer = layer_id, color = color)

    def add_dependency(self, task_id1, task_id2, id):
        """
        Adds a dependency (edge) to the graph.

        Parameters
        ----------
        task_id1 : int
            The id of the source task.
        task_id2 : int
            The id of the destination task.
        """
        self.graph.add_edge(task_id1, task_id2, id = id)

    def add_task_fully_qualified(self, id, type, layer_id, ct_required, dep, color = "lightblue"):
        """
        Adds a task (node) to the graph. Differently from the add_task method, this method allows to 
        specify all the parameters of the task that will be used during the simulation as a dictionary,
        that is then appended to the list of nodes.

        Parameters
        ----------
        id : int
            The id of the task. If None, the id is automatically assigned.
        layer_id : int
            The layer id of the task.
        node: int
            The node to wich the task is assigned.
        type : str
            The type of the task.
        ct_required : float
            The computing time required by the task.
        dep : list
            The list of dependencies of the task.
        color : str
            The color of the task.

        """
        if id is None:
            id = self.id
            self.id += 1
        
        elem = {"id": id, "type": type, "layer_id": layer_id, "ct_required": ct_required, "dep": dep}
        self.graph.add_node(id, layer = layer_id, color = color)
        self.nodes[id] = elem



    def add_dependency_fully_qualified(self, task_id1, task_id2, id, type, size, pt_required, dep, cl = 0):
        """
        Adds a dependency (edge) to the graph. Differently from the add_dependency method, this method allows to
        specify all the parameters of the dependency that will be used during the simulation as a dictionary,
        that is then appended to the list of edges.

        Parameters
        ----------
        task_id1 : int
            The id of the source task.
        task_id2 : int
            The id of the destination task.
        id: int
            The id of the dependency. If None, the id is automatically assigned.
        type : str
            The type of the dependency.
        src : int
            The source PE of the dependency.
        dst : int
            The destination PE of the dependency.
        size : float
            The size of the data to be transferred.
        pt_required : float 
            The processing time required by the dependency.
        dep : list
            The list of dependencies of the dependency.
        """
        elem = {"id": id, "size": size, "dep": dep,"type": type, "cl" : cl, "pt_required": pt_required }
        self.graph.add_edge(task_id1, task_id2, id = id)
        self.edges[id] = elem
    

    def json_to_graph(self, structure):
        """
        Initializes the graph from a JSON-like list of elements, representing both
        the nodes and edges of the graph. Those elements are obtained as the output of
        a compile, whose job is to analyze the structure of the NN and perform the slicing
        of the layers.

        Parameters
        ----------
        structure : list
            A JSON-like list of elements, representing the nodes and edges of the graph.
        
        Returns
        -------
        None
        """

        NODES_TYPES = ["COMP_OP"]
        EDGES_TYPES = ["WRITE", "READ", "READ_REQ", "WRITE_REQ"]

        c_to_m = {}
        m_to_c = {}

        for elem in structure:
            if elem["type"] in NODES_TYPES:
                # append the node to the list
                self.nodes[elem["id"]] = elem 
                # delete mapping info if present
                if elem.get("node") is not None:
                    del elem["node"]
                self.graph.add_node(elem["id"], layer = elem["layer_id"], color = 'lightblue')
                for dep in elem["dep"]:
                    if m_to_c.get(dep) is None:
                        m_to_c[dep] = []
                    m_to_c[dep].append(elem["id"])
                

            elif elem["type"] in EDGES_TYPES:
                # append the edge to the list
                self.edges[elem["id"]] = elem
                # delete mapping info if present
                if elem.get("src") is not None:
                    del elem["src"]
                if elem.get("dst") is not None:
                    del elem["dst"]

                if elem["dep"][0] == -1:
                    assert len(elem["dep"]) == 1
                    if c_to_m.get("start") is None:
                        c_to_m["start"] = []
                    c_to_m["start"].append(elem["id"])
                else:
                    for dep in elem["dep"]:
                        if c_to_m.get(dep) is None:
                            c_to_m[dep] = []
                        c_to_m[dep].append(elem["id"])

        for node1 in c_to_m.keys():
            for edge in c_to_m[node1]:
                if  edge not in m_to_c.keys():
                    node2 = "end"
                    self.graph.add_edge(node1, node2, id = edge)
                else:
                    for node2 in m_to_c[edge]:
                        self.graph.add_edge(node1, node2, id = edge)

    def graph_to_json(self, verbose = False):
        """
        Returns the graph as a JSON-like list of elements.

        Parameters
        ----------
        nodes_mapping_info : dict
            A dictionary representing the mapping of the nodes onto the NoC grid:
            nodes_mapping_info = {node_id : node}
        edges_mapping_info : dict
            A dictionary representing the mapping of the edges onto the NoC grid:
            edges_mapping_info = {edge_id : {"src": src, "dst": dst}}
        verbose : bool
            If True, print the nodes and edges of the graph when constructing the list.
        
        Returns
        -------
        list
            A JSON-like list of elements, representing the nodes and edges of the graph.
        """
        
        structure = []
        satisfied_dependencies = [-1]
        # Remove elements whose element appearing more than once
        nodes = self.get_nodes()
        edges = self.get_edges()
        
    
        while (len(nodes)>0 or len(edges)>0):
            for edge in edges:
                if set(edge["dep"]).issubset(satisfied_dependencies):
                    if self.edges_mapping:
                        edge["src"] = self.edges_mapping[edge["id"]]["src"]
                        edge["dst"] = self.edges_mapping[edge["id"]]["dst"]
                    structure.append(edge)
                    if verbose:
                        print("\t\t\t Appending edge: ", edge["id"])
                        print("====================================")
                        print(self.EDGE_INFO.format(edge["id"], edge["type"], edge["size"], edge["pt_required"], edge["dep"]))
                        if self.edges_mapping:
                            print("Mapping info: ")
                            print("src: ", self.edges_mapping[edge["id"]]["src"])
                            print("dst: ", self.edges_mapping[edge["id"]]["dst"])
                        print("====================================")
                    satisfied_dependencies.append(edge["id"])
                    edges.remove(edge)
                    break
            for node in nodes:
                # check that all the dependeices are satisfied
                if set(node["dep"]).issubset(satisfied_dependencies):
                    if self.nodes_mapping:
                        node["node"] = self.nodes_mapping[node["id"]]
                    structure.append(node)
                    if verbose:
                        print("\t\t\t Appending node: ", node["id"])
                        print("====================================")
                        print(self.NODE_INFO.format(node["id"], node["type"], node["layer_id"], node["ct_required"], node["dep"]))
                        if self.nodes_mapping:
                            print("Mapping info: ")
                            print("node: ", self.nodes_mapping[node["id"]])
                        print("====================================")
                    satisfied_dependencies.append(node["id"])
                    nodes.remove(node)
                    break
        assert len(nodes) == 0 and len(edges) == 0
        return structure
                                    
    def get_node(self, id,  verbose = False):
        """
        Returns the node with the given id.

        Parameters
        ----------
        id : int
            The id of the node to be returned.
        
        Returns
        -------
        dict
            The node with the given id.
        """

        node = self.nodes.get(id)

        if verbose:
            print("====================================")
            print(self.NODE_INFO.format(node["id"], node["type"], node["layer_id"], node["ct_required"], node["dep"]))
            print("====================================")

        return node
            
        return None
    
    def get_edge(self, id, verbose = False):
        """
        Returns the edge with the given id.

        Parameters
        ----------
        id : int
            The id of the edge to be returned.
        
        Returns
        -------
        dict
            The edge with the given id.
        """

        edge = self.edges.get(id)

        if verbose:
            print("====================================")
            print(self.EDGE_INFO.format(edge["id"], edge["type"], edge["size"], edge["pt_required"], edge["dep"]))
            print("====================================")
        return edge
            
        return None
    

    def get_nodes(self, verbose = False):
        """
        Returns the list of nodes.

        Parameters
        ----------
        None
        
        Returns
        -------
        list
            The list of nodes.
        """
        
        if verbose:
            print("------------------------------------")
            print("\t\t\tNODES (", len(self.nodes), ")")
            print("------------------------------------")
            for node in self.nodes.values():
                print("====================================")
                print(self.NODE_INFO.format(node["id"], node["type"], node["layer_id"], node["ct_required"], node["dep"]))
                print("====================================")


        return list(self.nodes.values())
    
    def get_edges(self, verbose = False):
        """
        Returns the list of edges.

        Parameters
        ----------
        None
        
        Returns
        -------
        dict
            The list of edges.
        """

        if verbose:
            print("------------------------------------")
            print("\t\t\tEDGES (", len(self.edges), ")")
            print("------------------------------------")
            for edge in self.edges.values():
                print("====================================")
                print(self.EDGE_INFO.format(edge["id"], edge["type"], edge["size"], edge["pt_required"], edge["dep"]))
                print("====================================")   


        return list(self.edges.values())
    
    def get_graph(self):
        """
        Returns the graph.

        Parameters
        ----------
        None
        
        Returns
        -------
        networkx.DiGraph
            The graph.
        """

        return self.graph
    
    def plot_graph(self, file_path = None):
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

        pos = nx.multipartite_layout(self.graph, subset_key="layer")
        colors = [self.graph.nodes[node]["color"] for node in self.graph.nodes()]
        labels = {node: node for node in self.graph.nodes()}
        node_size = 1000
        node_shape = "s"

        for node_id, node in enumerate(self.graph.nodes()):
            nx.draw_networkx_nodes(self.graph, pos, nodelist = [node], node_color = colors[node_id], node_size = node_size, node_shape = node_shape)
            nx.draw_networkx_labels(self.graph, pos, labels = labels)
        nx.draw_networkx_edges(self.graph, pos, node_size = node_size, node_shape = node_shape)
        plt.tight_layout()
        plt.axis("off")
        
        if file_path is not None:
            file_path = os.path.join(os.path.dirname(__file__), file_path)
            plt.savefig(file_path)
        plt.show()


