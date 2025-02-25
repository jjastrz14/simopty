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
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
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
        self.points[2] = []  # Reconfiguring memories NVMs
        self.artists_points = {} # List of the artists for the points
        self.artists_points[0] = [] 
        self.artists_points[1] = []
        self.artists_points[2] = []
        self.artists_points["txt"] = [] 
        self.connections = [] # List of the connection between the points 
        self.artists_hconnections = {} # List of the artists for the connections (horizontal)
        self.artists_reconf_connections = {} # List of the artists for the connections between the reconfiguring memories and the PEs
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
        self.topology = arch["topology"]
        self.k = arch["k"]
        self.n = arch["n"]
        self.reconf = arch["reconfiguration"]

        # Number of layers
        self.num_of_layers = 2  # one layer for NoC elements, the other for NPUs + one for reconfiguring memories
        if self.reconf != 0:
            self.num_of_layers += 1
        proc_elemnt_ids = []
        for pe in range(arch["k"]** arch["n"]):
            proc_elemnt_ids.append(pe)


        # Points is a list of tuples
        for p in range(arch["k"]** arch["n"]):
            x = p % arch["k"]
            y = p // arch["k"]
            z = 0
            self.points[0].append((x, y, z)) # 0 is for NoC elements
            self.points[1].append((x, y, z+0.5)) # 1 is for NPUs
            # if the field "reconfiguration" is set to any number different from 0,
            # we must also append a third layer representing the reconfiguring memories
            if self.reconf != 0:
                self.points[2].append((x, y, z+1))

        # Build the list of connections based on the mesh topology
        
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

        if self.reconf != 0:
            x.append(self.points[1][p1_ix][0])
            x.append(self.points[2][p2_ix][0])

            y.append(self.points[1][p1_ix][1])
            y.append(self.points[2][p2_ix][1])

            z.append(self.points[1][p1_ix][2])
            z.append(self.points[2][p2_ix][2])

            artist, = self.ax.plot(x, y, z, color='black', alpha = 0.3)
            self.artists_reconf_connections[(p2_ix, p2_ix)] = artist


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
        """
        Plot the connections between the nodes/points
        """
        for p in range(len(self.points[0])):
            self.vertical_connection(p, p)

        for c in self.connections:
            p1_ix, p2_ix = c[0]
            in_plane= c[1]

            self.horizontal_connection(p1_ix, p2_ix, in_plane)
    ###############################################################################


    def annotate_points(self):
        """
        Annotating the points using their index
        """
        points_coordinates = np.array(self.points[1])
        i = 0
        for x, y, z in zip(points_coordinates[:, 0], points_coordinates[:, 1], points_coordinates[:, 2]):
            self.artists_points["txt"].append(self.ax.text(x, y, z + 0.57 , i, size=8, color='k', fontdict={'weight': 'bold'}, ha='left', va='bottom'))
            i = i + 1

    def plot_nodes(self, points):
        """
        Annotating the points (NoC Nodes) using their index
        """
        points_coordinates = []
        for p in points:
            points_coordinates.append(p)
            x = p[0]
            y = p[1]
            z = p[2]
            artist, = self.ax.plot(x, y, z, color = "lightseagreen", marker="o", markersize=10, alpha = 0.3)
            self.artists_points[0].append(artist)
        # points_coordinates = np.array(points_coordinates)
        # xs = points_coordinates[:, 0]
        # ys = points_coordinates[:, 1]
        # zs = points_coordinates[:, 2]
        # self.artists_points[0].append(self.ax.scatter(xs, ys, zs,color = "lightseagreen", s = 200, alpha = 0.3))#, marker=m)


    def plot_pes(self, points):
        """
        Annotating the points (PEs) using their index
        """
        points_coordinates = []
        for p in points:
            points_coordinates.append(p)
            x = p[0]
            y = p[1]
            z = p[2]
            artist, = self.ax.plot(x, y, z, color = "tomato", marker="s", markersize=10, alpha = 0.3)
            self.artists_points[1].append(artist)
        # points_coordinates = np.array(points_coordinates)
        # xs = points_coordinates[:, 0]
        # ys = points_coordinates[:, 1]
        # zs = points_coordinates[:, 2]
        # self.artists_points[1].append(self.ax.scatter(xs, ys, zs,color = "tomato" , s = 200, marker="s", alpha = 0.3)) 

    def plot_reconf(self, points):
        """
        Annotating the points (Reconfiguring Memories) using their index
        """
        points_coordinates = []
        for p in points:
            points_coordinates.append(p)
            x = p[0]
            y = p[1]
            z = p[2]
            artist, = self.ax.plot(x, y, z, color = "limegreen", marker="D", markersize=10, alpha = 0.3)
            self.artists_points[2].append(artist)
        # points_coordinates = np.array(points_coordinates)
        # xs = points_coordinates[:, 0]
        # ys = points_coordinates[:, 1]
        # zs = points_coordinates[:, 2]
        # self.artists_points[2].append(self.ax.scatter(xs, ys, zs,color = "khaki" , s = 200, marker="D", alpha = 0.3))
    ###############################################################################

    def colorize_nodes(self, currently_active:set, verbose:bool = False):
        """
        Colorize the nodes
        """
        if verbose:
            print("Currently active nodes: ", currently_active)
        for i in range(len(self.points[0])):
            target_alpha = 1 if i in currently_active else 0.3
            if self.artists_points[0][i].get_alpha() != target_alpha:
                self.artists_points[0][i].set_alpha(target_alpha)

                
    def colorize_pes(self, currently_active_comp:set, currently_active_traf:set, currently_active_reconf: set, verbose: bool = False):
        """
        Colorize the PEs
        """
        if verbose:
            print("Currently active PEs for computation: ", currently_active_comp)
            print("Currently active PEs for traffic: ", currently_active_traf)
        for i in range(len(self.points[1])):
            if i in currently_active_comp:
                assert i not in currently_active_reconf
                self.artists_points[1][i].set_color("tomato")
                self.artists_points[1][i].set_alpha(1)
            elif i in currently_active_traf:
                self.artists_points[1][i].set_color("khaki")
                self.artists_points[1][i].set_alpha(0.8)
            elif i in currently_active_reconf:
                self.artists_points[1][i].set_color("limegreen")
                self.artists_points[1][i].set_alpha(1)
            else:
                if self.artists_points[1][i].get_color() != "tomato" or self.artists_points[1][i].get_alpha() != 0.3:
                    self.artists_points[1][i].set_color("tomato")
                    self.artists_points[1][i].set_alpha(0.3)

    def colorize_reconf(self, currently_active:set, verbose: bool = False):
        """
        Colorize the reconfiguring memories
        """
        if verbose:
            print("Currently active reconfiguring memories: ", currently_active)
        for i in range(len(self.points[2])):
            if i in currently_active:
                self.artists_points[2][i].set_alpha(1)
            else:
                self.artists_points[2][i].set_alpha(0.3)

    def colorize_connections(self, currently_active:set, verbose: bool = False):
        """
        Colorize the connections
        """
        if verbose:
            print("Currently active connections: ", currently_active)
        for c in self.connections:
            to_check = self.artists_vconnections if c[0][0] == c[0][1] else self.artists_hconnections
            target_alpha = 1 if c[0] in currently_active else 0.3
            if isinstance(to_check[c[0]], list):
                for a in to_check[c[0]]:
                    if a.get_alpha() != target_alpha:
                        a.set_alpha(target_alpha)
            else:
                if isinstance(to_check[c[0]], list):
                    for a in to_check[c[0]]:
                        a.set_alpha(0.3)
                        
                else:
                    to_check[c[0]].set_alpha(0.3)

    def colorize_reconf_connections(self, currently_active:set, verbose: bool = False):
        """
        Colorize the connections between the reconfiguring memories and the PEs
        """
        if verbose:
            print("Currently active connections between reconfiguring memories and PEs: ", currently_active)
        #loop over the currently active connections, find the corresponding artists and set their alpha to 1
        for c in [(k,k) for k in range(len(self.points[0]))]:
            if c[0] in currently_active:
                if isinstance(self.artists_reconf_connections[c[0]], list):
                    for a in self.artists_reconf_connections[c[0]]:
                        a.set_alpha(1)
                else:
                    self.artists_reconf_connections[c[0]].set_alpha(1)
            else:
                if isinstance(self.artists_reconf_connections[c[0]], list):
                    for a in self.artists_reconf_connections[c[0]]:
                        a.set_alpha(0.3)
                else:
                    self.artists_reconf_connections[c[0]].set_alpha(0.3)

            
    

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

    def gen_activity_animation(self, logger, fps: int = 100, file_name: Union[str, None] = None, verbose: bool = True):
        """
        The function takes as input a logger object (defined in the nocsim module, exposed in restart). This is used as a chonological timeline
        on which the events that happen on NoC are registered. We unroll this timeline and plot it in a matplotlib animation

        Args:
        - logger: nocsim.EventLogger object
        - fine_name: a string, 

        Returns:
        - None
        """

        anti_events_map = { nocsim.EventType.IN_TRAFFIC : nocsim.EventType.OUT_TRAFFIC,
                            nocsim.EventType.END_COMPUTATION : nocsim.EventType.START_COMPUTATION,
                            nocsim.EventType.END_RECONFIGURATION : nocsim.EventType.START_RECONFIGURATION,
                            nocsim.EventType.END_SIMULATION : nocsim.EventType.START_SIMULATION}

        self.timeStamp = self.ax.text(0, 0, 1., 0, size=12, color='red', fontdict={'weight': 'bold'}, ha='left', va='bottom', transform=self.ax.transAxes)
        cycles = logger.events[-1].cycle
        events_pointer = 0 # pointer to the events in the logger
        current_events = set() # set of the current events

        def _init_graph():
            nonlocal events_pointer, current_events
            events_pointer = 0
            current_events = set()

        def _update_graph(cycle):
            nonlocal events_pointer, current_events
            
            # Process events for current cycle
            while events_pointer < len(logger.events) and cycle >= logger.events[events_pointer].cycle:
                event = logger.events[events_pointer]
                if event.type in anti_events_map.values():
                    current_events.add(events_pointer)
                else:
                    # Find and remove matching event
                    event_type = anti_events_map[event.type]
                    additional_info = event.additional_info
                    ctype = event.ctype
                    to_remove = next(
                        (e for e in current_events 
                        if logger.events[e].type == event_type 
                        and logger.events[e].additional_info == additional_info 
                        and logger.events[e].ctype == ctype),
                        None
                    )
                    if to_remove is not None:
                        current_events.remove(to_remove)
                events_pointer += 1

            # Prepare sets for active elements
            currently_active_nodes = set()
            currently_active_pes_comp = set()
            currently_active_pes_reconf = set()
            currently_active_pes_traf = set()
            currently_active_connections = set()

            # Single pass through current events to collect all active elements
            for event_idx in current_events:
                event = logger.events[event_idx]
                if event.type == nocsim.EventType.OUT_TRAFFIC:
                    for h in event.info.history:
                        if h.start <= cycle and h.end > cycle:
                            currently_active_nodes.add(h.rsource)
                            currently_active_nodes.add(h.rsink)
                            currently_active_connections.add(tuple(sorted([h.rsource, h.rsink])))
                            if h.rsource == h.rsink:
                                currently_active_pes_traf.add(h.rsource)
                            break
                        elif h.start > cycle:
                            currently_active_nodes.add(h.rsource)
                            break
                elif event.type == nocsim.EventType.START_COMPUTATION:
                    currently_active_pes_comp.add(event.info.node)

                elif logger.events[event_idx].type == nocsim.EventType.START_RECONFIGURATION:
                    currently_active_pes_reconf.add(event.additional_info)
                    

            self.colorize_nodes(currently_active_nodes, verbose)
            self.colorize_pes(currently_active_pes_comp, currently_active_pes_traf, currently_active_pes_reconf, verbose)
            self.colorize_connections(currently_active_connections, verbose)
            self.colorize_reconf(currently_active_pes_reconf, verbose)
            #self.colorize_reconf_connections(currently_active_pes_reconf, verbose)
            self.timeStamp.set_text(f"Cycle: {cycle}")

            #plt.draw()
        # Crea l'animazione utilizzando FuncAnimation
        ani = FuncAnimation(self.fig, 
                            _update_graph,
                            #blit = True, #to speed up the animation
                            frames=range(cycles), 
                            init_func=_init_graph ,repeat=False, 
                            interval=1000/fps
                            )

        # Salva l'animazione se file_name è specificato
        if file_name:
            ani.save(file_name, 
                     writer='pillow', 
                     fps=fps,
                     savefig_kwargs={'facecolor': 'white'},
                     dpi=100
                     )
            print(f"Animation saved to {file_name}")
            
            
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
        self.plot_reconf(self.points[2])
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
        self.node_events = {i: {"comp": [], "traf": [], "recon": []} for i in range(len(self.points[1]))}
    
        # Process computation events
        for event in logger.events:
            ######### COMPUTATIONS #######
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
            ######### TRAFFIC #######    
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
            ######### RECONFIGURATION #######
            elif event.type == nocsim.EventType.START_RECONFIGURATION:
                node = event.additional_info
                start = event.cycle
                
                # Find matching END_RECONFIGURATION with error handling
                try:
                    end_event = next(
                        e for e in logger.events 
                        if e.type == nocsim.EventType.END_RECONFIGURATION 
                        and e.additional_info== node
                        and e.cycle > start  # Ensure valid duration
                    )
                    duration = end_event.cycle - start # Inclusive duration already embeded in cycles from logger
                    self.node_events[node]["recon"].append((start, duration))
                except StopIteration:
                    print(f"Warning: No END_RECONFIGURATION found for node {node} at cycle {start}")
                    continue 
            ######### Events to omit #####
            elif event.type == nocsim.EventType.START_SIMULATION or nocsim.EventType.END_SIMULATION:
                continue
            else:
                raise TypeError(f"Unknown event type: {event.type}") 
                        
    
    def _print_node_events(self):
        """Print event data for debugging."""
        
        for node, events in self.node_events.items():
            print(f"Node {node}:")
            print(f"Computation events: {events['comp']}")
            print(f"Traffic events: {events['traf']}")
            print(f"Reconfiguration events: {events['recon']}")
            print()

    def plot_timeline(self, filename):
        """Draw horizontal bars for events."""
        
        # for node, events in self.node_events.items():
        #     # Plot computation events (red)
        #     if events["comp"]:
        #         self.ax2d.broken_barh(events["comp"], (node - 0.4, 0.8), facecolors='tomato', label="Computation")
        #     # Plot traffic events (blue)
        #     if events["traf"]:
        #         self.ax2d.broken_barh(events["traf"], (node - 0.4, 0.8), facecolors='dodgerblue', label="Traffic", alpha=0.5)
        #     # Plot reconfiguration events (green)
        #     if events["recon"]:
        #         self.ax2d.broken_barh(events["recon"], (node - 0.4, 0.8), facecolors='limegreen', label="Reconfiguration", alpha=0.7)
        #     #elif len(self.node_events.items()) == 0:
        #         #print((f"No reconfiguration events found for node {node}"))
        #     #else: 
        #         #raise RuntimeError(f"Something wrong with plot timeline function")
        
        # # Deduplicate legend entries
        # handles, labels = self.ax2d.get_legend_handles_labels()
        # unique_labels = dict(zip(labels, handles))
        # self.ax2d.legend(unique_labels.values(), unique_labels.keys())
        
        #(key, color, label, alpha)
        event_types = [
        ('comp', 'tomato', 'Computation', 1.0), 
        ('traf', 'dodgerblue', 'Traffic', 0.5),
        ('recon', 'limegreen', 'Reconfiguration', 0.7)
        ]
    
        # Track used labels to prevent duplicates in legend
        used_labels = set()

        for node, events in self.node_events.items():
            has_events = False  # Flag to check if node has any events
            
            for event_key, color, label, alpha in event_types:
                event_list = events.get(event_key, [])
                if event_list:
                    has_events = True
                    # Use label only if it hasn't been used before
                    current_label = label if label not in used_labels else None
                    self.ax2d.broken_barh(
                        event_list,
                        (node - 0.4, 0.8),
                        facecolors=color,
                        label=current_label,
                        alpha=alpha
                    )
                    if current_label:
                        used_labels.add(label)
            
            # Handle nodes with no events
            if not has_events:
                print(f"No events found for node {node}")
        
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
            print(f"Timeline graph saved to {filename}")





