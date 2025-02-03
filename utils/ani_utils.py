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
        
        # artists for 3D blitting
        self.artists = []

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
    ###############################################################################

    def plot_connections(self):
        """
        Plot all the connections (vertical and horizontal) as a single Line3DCollection.
        """
        segments = []  # Each segment is a list of two (x, y, z) tuples.
        colors   = []  # Color (with alpha) for each segment.

        # Vertical connections: connect corresponding points in points[0] and points[1]
        for p in range(len(self.points[0])):
            top = self.points[0][p]
            bottom = self.points[1][p]
            segments.append([top, bottom])
            colors.append((0, 0, 0, 0.3))  # black with alpha=0.3

        # Horizontal connections:
        for connection in self.connections:
            (p1_ix, p2_ix), in_plane = connection

            if in_plane == 0:
                # A standard horizontal connection within the same plane.
                p1 = self.points[0][p1_ix]
                p2 = self.points[0][p2_ix]
                segments.append([p1, p2])
                colors.append((0, 0, 0, 0.3))
            else:
                # A wrap-around connection, where we split the connection into two segments.
                # border is fixed
                border = 0.5
                p1 = self.points[0][p1_ix]
                p2 = self.points[0][p2_ix]

                # Determine the offset for the first segment (from p1 to the border)
                if in_plane == 1:
                    offset1 = (-border, 0, 0)
                elif in_plane == 2:
                    offset1 = (border, 0, 0)
                elif in_plane == 3:
                    offset1 = (0, -border, 0)
                elif in_plane == 4:
                    offset1 = (0, border, 0)
                else:
                    offset1 = (0, 0, 0)  # Should not happen

                p1_border = (p1[0] + offset1[0],
                            p1[1] + offset1[1],
                            p1[2] + offset1[2])

                # For p2 we use the opposite offset so that the two segments point to the same border.
                if in_plane == 1:
                    offset2 = (border, 0, 0)
                elif in_plane == 2:
                    offset2 = (-border, 0, 0)
                elif in_plane == 3:
                    offset2 = (0, border, 0)
                elif in_plane == 4:
                    offset2 = (0, -border, 0)
                else:
                    offset2 = (0, 0, 0)

                p2_border = (p2[0] + offset2[0],
                            p2[1] + offset2[1],
                            p2[2] + offset2[2])

                segments.append([p1, p1_border])
                segments.append([p2, p2_border])
                colors.append((0, 0, 0, 0.4))
                colors.append((0, 0, 0, 0.4))

        # Create the 3D line collection with all segments and add it to the axes.
        self.connection_lines = Line3DCollection(segments, colors=colors, linewidths=1)
        self.ax.add_collection(self.connection_lines)
        # for animation
        self.artists.append(self.connection_lines)

    ###############################################################################


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
        """
        Annotating the points (NoC Nodes) using their index
        """
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]
        
        self.node_dots = self.ax.scatter(
            xs, ys, zs, 
            color="lightseagreen", 
            s=100, 
            alpha=0.3,
            marker="o"
        )
        self.artists.append(self.node_dots)


    def plot_pes(self, points):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]
        
        self.pe_comp_dots = self.ax.scatter(
            [], [], [],  # Empty initially
            color="tomato", 
            s=100, 
            alpha=0.3,
            marker="s"
        )
        self.pe_traf_dots = self.ax.scatter(
            [], [], [],
            color="khaki",
            s=100,
            alpha=0.8,
            marker="D"
        )
        self.artists.extend([self.pe_comp_dots, self.pe_traf_dots])
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
    
    def _init(self):
        """Initialize blitting - draw static elements once"""
        self.timeStamp = self.ax.text(
            0.5, 0.9, 0.5, 
            "", 
            transform=self.ax.transAxes,
            ha = 'center')
        self.artists.append(self.timeStamp)
        
        return self.artists  
            

    def gen_activity_animation(self, logger, pause: float = 0.5, file_name: Union[str, None] = None, verbose: bool = False):
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
                            nocsim.EventType.END_SIMULATION : nocsim.EventType.START_SIMULATION
                            }
        cycles = logger.events[-1].cycle

        # ===== Precomputation Loop =====
        event_activations = []
        events_pointer = 0 # pointer to the events in the logger
        current_events = set() # set of the current events

        for cycle in range(cycles + 1):
            #loop to filer events that are active at the current cycle
            while events_pointer < len(logger.events):
                current_event = logger.events[events_pointer]
                
                if current_event.cycle > cycle:
                    break  # Stop when future events
                
                if current_event.type in anti_events_map.values():
                    current_events.add(events_pointer)
                    events_pointer += 1
                else:
                    # Find matching start event
                    event_to_remove = None
                    for ev in current_events:
                        if (logger.events[ev].type == anti_events_map[current_event.type] and
                            logger.events[ev].additional_info == current_event.additional_info and
                            logger.events[ev].ctype == current_event.ctype):
                            event_to_remove = ev
                            break
                    
                    if event_to_remove is not None:
                        current_events.remove(event_to_remove)
                    events_pointer += 1

            # ----- Calculate active elements -----
            active_nodes = set()
            active_conns = set()
            active_pes_comp = set()
            active_pes_traf = set()
            
            #loop to collect data of the events
            for event_idx in current_events:
                
                event = logger.events[event_idx]
                
                if event.type == nocsim.EventType.OUT_TRAFFIC:
                    # check what type of channel is active at the moment
                    for h in event.info.history:
                        # find the h element such that h[3]>= cycle and h4 <= cycle
                        if h.start <= cycle and h.end > cycle:
                            connection = tuple(sorted([h.rsource, h.rsink]))
                            active_conns.add(connection)
                            active_nodes.update([h.rsource, h.rsink])
                            if h.rsource == h.rsink:
                                active_pes_traf.add(h.rsource)
                            break
                        elif h.start > cycle:
                            # the connection is not active, the packet is still being processed by the source
                            active_nodes.add(h.rsource)
                            break   

                elif event.type == nocsim.EventType.START_COMPUTATION:
                    active_pes_comp.add(event.info.node)
                    
            event_activations.append((active_nodes, active_conns, active_pes_comp, active_pes_traf))
                
        def _update_graph(cycle):
            """Optimized update function using precomputed data and blitting"""
            # Get precomputed data for this cycle
            active_nodes, active_conns, pes_comp, pes_traf = event_activations[cycle]
            
            # Update NoC nodes (array-based)
            node_alphas = np.full(len(self.points[0]), 0.3)  # Default alpha
            node_alphas[list(active_nodes)] = 1.0
            self.node_dots.set_alpha(node_alphas)
            
            self.connection_lines.set_alpha(1.0)

            # Update PE markers (computation vs traffic)
            if pes_comp:
                comp_coords = np.array([self.points[1][idx] for idx in pes_comp])
                # comp_coords should have shape (N, 3) if not empty
                self.pe_comp_dots.set_offsets(comp_coords[:, :2])  # X,Y only for 3D
                self.pe_comp_dots.set_3d_properties(comp_coords[:, 2], 'z')  # Z-axis
            else:
                # Pass an empty 2D array with shape (0, 2) if no data exists
                self.pe_comp_dots.set_offsets(np.empty((0, 2)))
            
            # For traffic dots:
            if pes_traf:
                traf_coords = np.array([self.points[1][idx] for idx in pes_traf])
                self.pe_traf_dots.set_offsets(traf_coords[:, :2])
                self.pe_traf_dots.set_3d_properties(traf_coords[:, 2], 'z')
            else:
                self.pe_traf_dots.set_offsets(np.empty((0, 2)))
            
            # Update cycle counter text
            self.timeStamp.set_text(f"Cycle: {cycle}")
            
            # Return ALL modified artists (critical for blitting)
            return [
                self.node_dots,
                self.connection_lines,
                self.pe_comp_dots,
                self.pe_traf_dots,
                self.timeStamp
            ]

        # Crea l'animazione utilizzando FuncAnimation
        ani = FuncAnimation(self.fig, 
                            _update_graph,
                            init_func = self._init, 
                            blit = True, #to speed up the animation
                            frames=range(cycles), 
                            repeat=False, 
                            interval=pause*100
                            )

        # Salva l'animazione se file_name è specificato
        if file_name:
            ani.save(file_name, 
                     writer='pillow', 
                     fps=1/pause,
                     savefig_kwargs={'facecolor': 'white'},
                     dpi=150
                     )

        #plt.show()
            
            
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
        self.plot_nodes(self.points[0])
        self.plot_pes(self.points[1])
        self._init() #init for blitting
        self.gen_activity_animation(logger, pause, file_name, verbose = False)
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
            self.fig.savefig(filename, dpi=300)





