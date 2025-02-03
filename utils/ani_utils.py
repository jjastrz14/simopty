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
from collections import defaultdict
import json
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection, LineCollection
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
base_path = os.path.expanduser("~/Projects/restart/lib")
if not os.path.exists(base_path):
    raise FileNotFoundError(f"The required library path does not exist: {base_path}. Please update the PATH_TO_SIMULATOR variable with the correct path.")
PATH_TO_SIMULATOR = base_path
sys.path.append(PATH_TO_SIMULATOR)
import nocsim

class NoCPlotter:

    def __init__(self):

        self.fig = None # Figure Object
        self.ax = None # Axes Object
        self.topology = None # The topology of the mesh
        self.points = {} # List of the nodes/points
        self.points[0] = []  # NoC elements
        self.points[1] = []  # NPUs
        self.points[2] = []  # NVMs (for future use)
        self.artists_points = {} # List of the artists for the points
        self.artists_points[0] = [] 
        self.artists_points[1] = []
        self.artists_points["txt"] = [] 
        self.connections = [] # List of the connection between the points 
        self.artists_hconnections = {} # List of the artists for the connections (horizontal)
        self.artists_vconnections = {} # List of the artists for the connections (vertical)
        self.num_of_layers = 0
        self.layers = [] # list of the layers
        self.faces = [] # List of the faces, for drawing reasons
    

    def init(self, config_file):
        """
        Initialize the variables
        """

        def _get_neighbors(k,n,i, topology):
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
                if i % k != 0: # left neighbor
                    neighbors.append((i-1, 0))
                else:
                    neighbors.append((k-1+i, 1)) # 1 is for left wrap-around
                if i % k != k-1: # right neighbor
                    neighbors.append((i+1, 0))
                else:
                    neighbors.append((i-k+1, 2)) # 2 is for right wrap-around
                if i // k != 0: # bottom neighbor
                    neighbors.append((i-k, 0))
                else:
                    neighbors.append((k*(k-1)+i, 3)) # 3 is for bottom wrap-around
                if i // k != k-1: # top neighbor
                    neighbors.append((i+k, 0))
                else:
                    neighbors.append((i % k, 4)) # 4 is for top wrap-around
                
            else:
                raise RuntimeError('topology type is not supported')
            return neighbors
        
        # read the arch field from configuration file
        config = json.load(open(config_file))
        arch = config['arch']

        # Number of layers
        self.num_of_layers = 2 # one layer for NoC elements, the other for NPUs
        proc_elemnt_ids = []
        for pe in range(arch["k"]** arch["n"]):
            proc_elemnt_ids.append(pe)


        # Points is a list of tuples
        for p in range(arch["k"]** arch["n"]):
            x = p % arch["k"]
            y = p // arch["k"]
            z = 0
            self.points[0].append((x, y, z)) # 0 is for NoC elements
            self.points[1].append((x, y, z-0.5)) # 1 is for NPUs

        # Build the list of connections based on the mesh topology
        self.topology = arch["topology"]
        self.k = arch["k"]
        self.n = arch["n"]
        for node in range(arch["k"]** arch["n"]):
            neighbors = _get_neighbors(arch["k"], arch["n"], node, self.topology) +[(node, 0)]
            for neighbor in neighbors:
                sorted_connection = sorted([node, neighbor[0]])
                sorted_connection = (tuple(sorted_connection), neighbor[1])
                # check that no other connection with the same vertices is already in the list
                pcheck = False
                for c in self.connections:
                    if c[0] == sorted_connection[0]:
                        pcheck = True
                        break
                if not pcheck:
                    self.connections.append(sorted_connection)
        
    ###############################################################################
    def create_fig(self):
        """
        Create the figure object
        """
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_box_aspect([1*self.k, 1*self.k, self.k * 0.5])
        self.ax.axis("off")
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.timeStamp = self.ax.text(0, 0, 0.5, "Cycle: 0", size=12, color='red')
    ###############################################################################

    def vertical_connection(self, p1_ix, p2_ix):
        """
        Draws the vertical connection of the points
        """
        x = []
        y = []
        z = []

        x.append(self.points[0][p1_ix][0])
        x.append(self.points[1][p2_ix][0])

        y.append(self.points[0][p1_ix][1])
        y.append(self.points[1][p2_ix][1])

        z.append(self.points[0][p1_ix][2])
        z.append(self.points[1][p2_ix][2])

        artist, = self.ax.plot(x, y, z, color='black', alpha = 0.3)
        self.artists_vconnections[(p1_ix, p2_ix)] = artist


    def horizontal_connection(self, p1_ix, p2_ix, in_plane):
        """
        Draws the horizontal connection of the points
        """
        x = []
        y = []
        z = []
    
        if in_plane == 0:
            x.append(self.points[0][p1_ix][0])
            x.append(self.points[0][p2_ix][0])

            y.append(self.points[0][p1_ix][1])
            y.append(self.points[0][p2_ix][1])

            z.append(self.points[0][p1_ix][2])
            z.append(self.points[0][p2_ix][2])

            # plot only if the connection has not been plotted yet
            artist, = self.ax.plot(x, y, z, color='black', alpha = 0.3)
            self.artists_hconnections[(p1_ix, p2_ix)] = artist

        else:
            # if the in_plane flag is 0 (this can only happen in the torus topology)
            # this means that the considered connection is a wrap-around connection:
            # we represent this by drawing a line from the two points to the border of the mesh

            border = 0.5

            segment1 = [(self.points[0][p1_ix][0], self.points[0][p1_ix][1], self.points[0][p1_ix][2]), 
                        (self.points[0][p1_ix][0] + (- border if in_plane == 1 else border if in_plane == 2 else 0), 
                         self.points[0][p1_ix][1] + (-border if in_plane == 3 else  border if in_plane == 4 else 0), 
                         self.points[0][p1_ix][2])]
            segment2 = [(self.points[0][p2_ix][0], self.points[0][p2_ix][1], self.points[0][p2_ix][2]), 
                        (self.points[0][p2_ix][0] + ( border if in_plane == 1 else - border if in_plane == 2 else 0), 
                         self.points[0][p2_ix][1] + (border if in_plane == 3 else - border if in_plane == 4 else 0), 
                         self.points[0][p2_ix][2])]

            artist1, = self.ax.plot([segment1[0][0], segment1[1][0]], [segment1[0][1], segment1[1][1]], 
                                    [segment1[0][2], segment1[1][2]], color='black', alpha = 0.4)
            artist2, = self.ax.plot([segment2[0][0], segment2[1][0]], [segment2[0][1], segment2[1][1]], 
                                    [segment2[0][2], segment2[1][2]], color='black', alpha = 0.4)

            self.artists_hconnections[(p1_ix, p2_ix)] = [artist1, artist2]


    def plot_connections(self):
        segments = []
        self.connection_indices = defaultdict(list)  # Maps (node_pair) -> segment indices
        
        # Rebuild coordinates the same way as original connection plotting
        for idx, conn in enumerate(self.connections):
            # conn should be in format: 
            # (connection_type, (node1, node2), [connection_specific_params])
            conn_type = conn[0]
            node_pair = tuple(sorted(conn[1]))
            p1_ix, p2_ix = conn[1]
            
            # Regenerate coordinates using your original connection logic
            if conn_type == "vertical":
                # Get coordinates from your original vertical connection logic
                x = self.points[0][p1_ix][0]  # X from router point
                y_start = self.points[0][p1_ix][1]
                y_end = self.points[0][p2_ix][1]
                segments.append([(x, y_start), (x, y_end)])
                
            elif conn_type == "horizontal":
                # Get coordinates from your original horizontal connection logic
                plane = conn[2]  # in_plane parameter
                y = self.points[0][p1_ix][1] + (0.1 if plane else -0.1)
                x_start = self.points[0][p1_ix][0]
                x_end = self.points[0][p2_ix][0]
                segments.append([(x_start, y), (x_end, y)])
            
            # Store mapping from logical connection to segment index
            self.connection_indices[node_pair].append(idx)
        
        # Create LineCollection with proper coordinates
        self.connection_colors = np.zeros((len(segments), 4))
        self.connection_colors[:, 3] = 0.3  # Default alpha
        self.line_collection = LineCollection(
            segments,
            colors=self.connection_colors,
            linewidths=2
        )
        self.ax.add_collection(self.line_collection)
        
    #########################


    def annotate_points(self):
        """
        Annotating the points using their index
        """
        points_coordinates = np.array(self.points[1])
        i = 0
        for x, y, z in zip(points_coordinates[:, 0], points_coordinates[:, 1], points_coordinates[:, 2]):
            self.artists_points["txt"].append(self.ax.text(x, y, z - 0.07, i, size=5, color='k', fontdict={'weight': 'bold'}, ha='left', va='bottom'))
            i = i + 1

    def plot_nodes(self, points):
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        self.node_colors = np.zeros((len(points), 4))
        self.node_colors[:] = (*plt.cm.Blues(0.6)[:3], 0.3)
        self.scatter_nodes = self.ax.scatter(x, y, s=100, color=self.node_colors)


    def plot_pes(self, points):
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        self.pe_colors = np.zeros((len(points), 4))
        self.pe_colors[:] = np.array([*plt.cm.Reds(0.6)[:3], 0.3])
        self.scatter_pes = self.ax.scatter(x, y, s=100, color=self.pe_colors)
    ###############################################################################

    def colorize_nodes(self, currently_active:set, verbose:bool = False):
        """
        Colorize the nodes
        """
        alphas = np.full(len(self.node_colors), 0.3)
        alphas[list(currently_active)] = 1.0
        self.node_colors[:, 3] = alphas
        self.scatter_nodes.set_facecolors(self.node_colors)

                
    def colorize_pes(self, active_comp, active_traf):
        # Reset to default
        self.pe_colors[:] = np.array([*plt.cm.Reds(0.6)[:3], 0.3])
        # Active computation
        self.pe_colors[list(active_comp), 3] = 1.0
        # Active traffic
        self.pe_colors[list(active_traf), :3] = plt.cm.YlOrBr(0.6)[:3] 
        self.pe_colors[list(active_traf), 3] = 0.8
        self.scatter_pes.set_facecolors(self.pe_colors)
            
    def colorize_connections(self, currently_active):
        active_indices = []
        for conn in currently_active:
            # Get all segment indices for this logical connection
            active_indices.extend(self.connection_indices.get(conn, []))
        # Update alphas
        self.connection_colors[:, 3] = 0.3
        if active_indices:
            self.connection_colors[active_indices, 3] = 1.0
        self.line_collection.set_colors(self.connection_colors)

    ###############################################################################
    def create_faces(self):
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

        for i in range(0, self.num_of_layers):
            layer = []

            for p in self.points[i]:
                    layer.append(p)
                    x_s.append(p[0])
                    y_s.append(p[1])
                    if (p[2] not in z_s):
                        z_s.append(p[2])

            self.layers.append(layer)

        # Making faces, only out of the corner points of the layer
        if self.topology == "mesh":
            for i in range(0, self.num_of_layers):
                x_min = min(x_s)
                x_max = max(x_s)
                y_min = min(y_s)
                y_max = max(y_s)
                face = list([(x_min, y_min, z_s[i]), (x_max, y_min, z_s[i]), (x_max, y_max, z_s[i]), (x_min, y_max, z_s[i])])
                self.faces.append(face)
        elif self.topology == "torus":
            added_border = 0.5
            for i in range(0, self.num_of_layers):
                x_min = min(x_s) - added_border
                x_max = max(x_s) + added_border
                y_min = min(y_s) - added_border
                y_max = max(y_s) + added_border
                face = list([(x_min, y_min, z_s[i]), (x_max, y_min, z_s[i]), (x_max, y_max, z_s[i]), (x_min, y_max, z_s[i])])
                self.faces.append(face)
    ###############################################################################


    def plot_faces(self):
        """
        Plot the faces
        """
        # Create multiple 3D polygons out of the faces
        poly = Poly3DCollection(self.faces, linewidths=1, alpha=0.1)
        faces_colors = []
        # Generating random color for each face
        for i in range(0, self.num_of_layers):
            red = np.random.randint(0, 256)
            green = np.random.randint(0, 256)
            blue = np.random.randint(0, 256)
            color = (red, green, blue)
            if color not in faces_colors:
                faces_colors.append('#%02x%02x%02x' % color)
        poly.set_facecolors(faces_colors)
        self.ax.add_collection3d(poly)
        
    ###############################################################################
    
    def precompute_activity(self, logger):
        max_cycle = logger.events[-1].cycle
        self.activity_data = [{'nodes': set(), 'pes_comp': set(), 'pes_traf': set(), 'connections': set()} for _ in range(max_cycle + 1)]
        
        pending_comp = {}  # (additional_info, ctype) -> (start_cycle, node)
        pending_traffic = {}  # (additional_info, ctype) -> (start_cycle, info)

        for event in logger.events:
            if event.type == nocsim.EventType.START_COMPUTATION:
                key = (event.additional_info, event.ctype)
                pending_comp[key] = (event.cycle, event.info.node)
                
            elif event.type == nocsim.EventType.END_COMPUTATION:
                key = (event.additional_info, event.ctype)
                if key in pending_comp:
                    start_cycle, node = pending_comp.pop(key)
                    for cycle in range(start_cycle, event.cycle):
                        if cycle <= max_cycle:
                            self.activity_data[cycle]['pes_comp'].add(node)
                            
            elif event.type == nocsim.EventType.OUT_TRAFFIC:
                key = (event.additional_info, event.ctype)
                pending_traffic[key] = (event.cycle, event.info)
                
            elif event.type == nocsim.EventType.IN_TRAFFIC:
                
                key = (event.additional_info, event.ctype)
                if key in pending_traffic:
                    start_cycle, info = pending_traffic.pop(key)
                    for cycle in range(start_cycle, event.cycle):
                        if cycle > max_cycle:
                            continue
                        active_in_source = True
                        for h in info.history:
                            if h.start <= cycle < h.end:
                                conn = tuple(sorted([h.rsource, h.rsink]))
                                self.activity_data[cycle]['connections'].add(conn)
                                self.activity_data[cycle]['nodes'].update([h.rsource, h.rsink])
                                if h.rsource == h.rsink:
                                    self.activity_data[cycle]['pes_traf'].add(h.rsource)
                                active_in_source = False
                                break
                            elif h.start > cycle:
                                break
                        if active_in_source:
                            self.activity_data[cycle]['nodes'].add(info.source)

    def gen_activity_animation(self, logger, pause: float = 0.5, file_name: Union[str, None] = None, verbose: bool = False):
        
        self.precompute_activity(logger)
        cycles = len(self.activity_data)

        def _update_graph(cycle):
            data = self.activity_data[cycle]
            self.colorize_nodes(data['nodes'])
            self.colorize_pes(data['pes_comp'], data['pes_traf'])
            self.colorize_connections(data['connections'])
            self.timeStamp.set_text(f"Cycle: {cycle}")
            return [self.scatter_nodes, self.scatter_pes, self.line_collection, self.timeStamp]

        ani = FuncAnimation(fig = self.fig, 
                            func = _update_graph,
                            blit = True, #to speed up the animation
                            frames = range(cycles), 
                            repeat = False, 
                            interval=pause*100
                            )
        if file_name:
            ani.save(file_name, 
                     writer='pillow', 
                     fps=1/pause,
                     savefig_kwargs={'facecolor': 'white'},
                     dpi=100
                     )
            
    ###############################################################################

    def plot(self,logger, pause, network_file = None, file_name = None, verbose = False):
        """
        Main Execution Point
        """
        curdir = os.path.dirname(__file__)
        network_file = curdir +'/../config_files/arch.json' if network_file is None else network_file

        try:
            network_file = sys.argv[1]
        except IndexError:
            pass
        self.init(network_file)
        self.create_fig()
        self.plot_connections()
        self.annotate_points()
        # create_faces()
        # plot_faces()
        self.plot_nodes(self.points[0])
        self.plot_pes(self.points[1])
        self.gen_activity_animation(logger, pause,file_name, verbose)
        #plt.show()
    ###############################################################################


class NoCTimelinePlotter(NoCPlotter):
    """Subclass for 2D timeline visualization. Inherits all NoCPlotter methods."""
    
    def __init__(self):
        super().__init__()
        self.fig2d = None  # Separate figure for 2D timeline
        self.ax2d = None
        self.node_events = {}  # Stores event data: {node: {"comp": [(start, duration)], "traf": [...]}}

    def setup_timeline(self, logger, config_file):
        """Initialize 2D timeline figure and preprocess events."""
        # Initialize parent class with architecture config
        self.init(config_file)  # Loads node data from config_file
        self._preprocess_events(logger)
        self.fig2d, self.ax2d = plt.subplots(figsize=(10, 6))
        self.ax2d.set_xlabel("Cycle")
        self.ax2d.set_ylabel("Node")
        self.ax2d.grid(False)
        self.max_cycle = logger.events[-1].cycle

    def _preprocess_events(self, logger):
        """Extract computation/traffic events per node."""
        self.node_events = {i: {"comp": [], "traf": []} for i in range(len(self.points[1]))}
    
        # Process computation events
        for event in logger.events:
            if event.type == nocsim.EventType.START_COMPUTATION:
                node = event.info.node
                start = event.cycle
                
                # Find matching END_COMPUTATION with error handling
                try:
                    end_event = next(
                        e for e in logger.events 
                        if e.type == nocsim.EventType.END_COMPUTATION 
                        and e.info.node == node
                        and e.cycle > start  # Ensure valid duration
                    )
                    duration = end_event.cycle - start + 1  # Inclusive duration
                    self.node_events[node]["comp"].append((start, duration))
                except StopIteration:
                    print(f"Warning: No END_COMPUTATION found for node {node} at cycle {start}")
                    continue 
                
            elif event.type == nocsim.EventType.OUT_TRAFFIC:
                id_message = event.additional_info
                communication_type = event.ctype #comunication type of the event
                start = event.cycle
                originating_node = event.info.history[0].rsource
                
                # Find matching IN_TRAFFIC with error handling
                try:
                    end_event = next(
                        e for e in logger.events 
                        if e.type == nocsim.EventType.IN_TRAFFIC 
                        and e.additional_info == id_message
                        and e.ctype == communication_type
                        and e.cycle > start  # Ensure valid duration
                    )
                    duration = end_event.cycle - start + 1  # Inclusive duration
                    
                    # Use the ORIGINATING NODE (first history entry's rsource) as the node ID
                    self.node_events[originating_node]["traf"].append((start, duration))
                except StopIteration:
                    print(f"Warning: No IN_TRAFFIC  found for id message {id_message}  with communication type {communication_type} at cycle {start}")
                    continue 
                        
    
    def _print_node_events(self):
        """Print event data for debugging."""
        
        for node, events in self.node_events.items():
            print(f"Node {node}:")
            print(f"Computation events: {events['comp']}")
            print(f"Traffic events: {events['traf']}")
            print()

    def plot_timeline(self, filename):
        """Draw horizontal bars for events."""
        
        for node, events in self.node_events.items():
            # Plot computation events (red)
            if events["comp"]:
                self.ax2d.broken_barh(events["comp"], (node - 0.4, 0.8), facecolors='tomato', label="Computation")
            # Plot traffic events (blue)
            if events["traf"]:
                self.ax2d.broken_barh(events["traf"], (node - 0.4, 0.8), facecolors='dodgerblue', label="Traffic", alpha=0.5)
        
        # Deduplicate legend entries
        handles, labels = self.ax2d.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        self.ax2d.legend(unique_labels.values(), unique_labels.keys())
        
        # Set y-ticks to node IDs
        self.ax2d.set_yticks(range(len(self.points[1])))
        #for debug purposes, only show first 2 nodes
        #self.ax2d.set_yticks(range(2))
        
        # set x-ticks to cycle numbers
        self.ax2d.set_xlim(0, self.max_cycle)
        # Auto-adjust ticks
        self.ax2d.xaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
        self.ax2d.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        
        # Add vertical lines at each major x-tick
        for tick in self.ax2d.xaxis.get_major_locator().tick_values(self.ax2d.get_xlim()[0], self.ax2d.get_xlim()[1]):
            self.ax2d.axvline(x=tick, color='grey', linestyle='-', linewidth=0.5, zorder = 0)

        # Add horizontal lines at the corners of the x nodes
        for node in range(len(self.points[1])):
            self.ax2d.axhline(y=node - 0.4, color='grey', linestyle='--', linewidth=0.5)
            self.ax2d.axhline(y=node + 0.4, color='grey', linestyle='--', linewidth=0.5)

        if filename: 
            self.fig2d.savefig(filename, dpi=300)





