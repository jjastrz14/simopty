'''
==================================================
File: optimizer.py
Project: simopty
File Created: Wednesday, 18th December 2024
Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''


"""
The optimizer.py module contains the classes used to perform the optimization of the mapping of a split-compiled NN onto a k-dimensional NoC grid.
"""

import os
import enum
import numpy as np
import simulator_stub as ss
import mapper as mp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from concurrent.futures import ThreadPoolExecutor
from dirs import *


class OptimizerType(enum.Enum):
    """
    Enum class representing the type of optimizer.
    """
    ACO = "ACO"


class OptimizationParameters:
    """
    Class representing the parameters of the optimization algorithm.
    In general, different optimizers may require different parameters.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class BaseOpt :
    def __init__(self, optimizer_type):
        self.optimizer_type = optimizer_type

    def __str__(self):
        return self.optimizer_type.value
    
    def run(self):
        pass
    

class AntColony(BaseOpt):
    def __init__(self, optimization_parameters, domain, task_graph):
        """
        
        Parameters:
        -----------
        optimization_parameters : OptimizationParameters
            The parameters of the optimization algorithm:
            - alpha : float
                The alpha parameter of the ACO algorithm.
            - beta : float
                The beta parameter of the ACO algorithm.
            - rho : float
                The evaporation rate of the pheromone.
            - n_ants : int
                The number of ants fro each colony
            - n_iterations : int
                The number of iterations of the algorithm.
            - n_best : int
                The number of best solutions to keep track of.

        domain : Grid
            The domain of the optimization problem, the NoC grid.
        task_graph : DepGraph
            The dependency graph of the tasks to be mapped onto the NoC grid.

        """

        super().__init__(OptimizerType.ACO)
        self.alpha = optimization_parameters.alpha # alpha parameter of the ACO algorithm
        self.beta = optimization_parameters.beta # beta parameter of the ACO algorithm
        self.rho = optimization_parameters.rho if hasattr(optimization_parameters, "rho") else None
        self.rho_start = optimization_parameters.starting_rho if hasattr(optimization_parameters, "starting_rho") else None
        self.rho_end = optimization_parameters.ending_rho if hasattr(optimization_parameters, "ending_rho") else None

        #self.q0 = optimization_parameters.q0 
        self.n_ants = optimization_parameters.n_ants
        self.n_iterations = optimization_parameters.n_iterations
        self.n_best = optimization_parameters.n_best

        # --- Domain and Task Graph ---
        self.domain = domain
        self.task_graph = task_graph

        # --- Pheromone and Heuristic Information ---
        # The DAG on which to optimize is made out of implementation points, particular combinations
        # of tasks and PEs, representig to which PE a task is mapped to.
        #  the number of nodes in this new graph = #tasks * #PEs
        # self.tau_start = np.zeros((domain.size))
        # self.tau_start[4] = 1
        self.tau_start = np.ones((domain.size))/ domain.size
        self.tau = np.ones((task_graph.n_nodes-1, domain.size, domain.size))/ (task_graph.n_nodes * domain.size * domain.size) # Pheromone matrix
        self.eta = np.ones((task_graph.n_nodes-1, domain.size, domain.size)) # Heuristic matrix

        tasks = [task["id"] for task in self.task_graph.get_nodes() if task["id"] != "start"]
        tasks.insert(0, "start")
        self.tasks = tasks


    def run_and_show_traces(self, single_iteration_func, **kwargs):
        """
        This function is used to show the most likely paths taken by the ants throughout the 
        execution of the algorithm.
        It produces a gif file, evaluating the pheromone traces at a certain iteration and 
        finally plotting those, then stiching them up to create the animation
        """

        fig, ax = plt.subplots()
        cmap = plt.get_cmap("magma")
        margin = 0.5
        ax.axis("off")
        ax.set_xlim( - 1 - margin, len(self.tasks) + margin)
        ax.set_ylim( - margin , self.domain.size + margin)

        a = 0.7 # ellipse width
        b = 0.5 # ellipse height    

        #implementation_points = []
        for i,task in enumerate(self.tasks[1:]):
            for j in range(self.domain.size):
                #implementation_points.append((i, j))
                ax.add_patch(patches.Ellipse((i, j), a, b, alpha = 1, facecolor = 'navajowhite', edgecolor = 'black', linewidth = 2, zorder = 2))
                ax.text(i, j, "(%s, %d)" % (task, j), ha = 'center', va = 'center', color = 'black')
        
        # Add starting and ending points
        ax.add_patch(patches.Ellipse((-1, int(self.domain.size/2)), a, b, alpha = 1, facecolor = 'palegreen', edgecolor = 'black', linewidth = 2, zorder = 2))
        ax.text(-1, int(self.domain.size/2), "start", ha = 'center', va = 'center', color = 'black', fontweight = 'bold')
        ax.add_patch(patches.Ellipse((len(self.tasks)-1, int(self.domain.size/2)), a, b, alpha = 1, facecolor = 'lightcoral', edgecolor = 'black', linewidth = 2, zorder = 2))
        ax.text(len(self.tasks)-1, int(self.domain.size/2), "end", ha = 'center', va = 'center', color = 'black', fontweight = 'bold')
        
        # Draw the connections (fully connected layers style)
        for i in range(len(self.tasks)-2):
            for j in range(self.domain.size):
                for k in range(self.domain.size):
                    ax.plot([i, i+1], [j, k], color = 'lightgrey', zorder = -2)

        for j in range(self.domain.size):
            ax.plot([-1, 0], [int(self.domain.size/2),j], color = 'lightgrey', zorder = -2)
            ax.plot([len(self.tasks)-2, len(self.tasks)-1], [j, int(self.domain.size/2)], color = 'lightgrey', zorder = -2)

        all_time_shortest_path = []
        edges = []

        def update(frame, edges = edges, all_time_shortest_path = all_time_shortest_path):
            # Each time the update function is called, an iteration of the algorithm is performed
            # and, if the frame is a multiple of the once_every parameter, the plot is actually updated

            all_time_shortest_path.append(single_iteration_func(frame, kwargs["once_every"], kwargs["rho_step"]))
            if len(all_time_shortest_path) > 1:
                assert len(all_time_shortest_path) == 2
                if all_time_shortest_path[1][1] < all_time_shortest_path[0][1]:
                    all_time_shortest_path[0] = all_time_shortest_path[1]
                all_time_shortest_path.pop(1)


            if frame % kwargs["once_every"] == 0:
                # extract the pheromone values and plot them on the corresponding edges
                

                for edge in edges:
                    edge.remove()
                edges.clear()

                for d_level in range(self.tau.shape[0]):
                    vmax = np.max(self.tau[d_level])
                    for i in range(self.tau.shape[1]):
                        for j in range(self.tau.shape[2]):
                            # find the maximum value out of the pheromones of che considered level
                            if self.tau[d_level, i, j] > 0:
                                edges.append(ax.plot([d_level, d_level+1], [i, j], color = cmap(self.tau[d_level, i, j]/vmax), zorder = self.tau[d_level, i, j]/vmax)[0])
                
                vmax_start = np.max(self.tau_start)
                for i in range(self.tau_start.shape[0]):
                    if self.tau_start[i] > 0:
                        edges.append(ax.plot([-1, 0], [int(self.domain.size/2), i], color = cmap(self.tau_start[i]/vmax_start), zorder = self.tau[d_level, i, j]/vmax)[0])

                return edges

            return []
                

        ani = animation.FuncAnimation(fig, update, frames = kwargs["n_iterations"], repeat = False) 
        plt.show()
        return all_time_shortest_path[0]

    def run(self, once_every = 10, show_traces = False):
        """
        Run the algorithm
        """

        def single_iteration(i, once_every, rho_step = 0):
            all_paths = self.generate_colony_paths()
            self.update_pheromones(all_paths)
            self.update_heuristics()
            shortest_path = min(all_paths, key=lambda x: x[1])
            moving_average = np.mean([path[1] for path in all_paths])
            if once_every is not None and i%once_every == 0:
                print("Iteration #", i, ", chosen path is:", shortest_path)
                print("Moving average for the path lenght is:", moving_average)
            
            self.evaporate_pheromones(rho_step)
            return shortest_path


        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)

        if self.rho_start is not None and self.rho_end is not None:
            self.rho = self.rho_start
            rho_step = (self.rho_end - self.rho_start) / self.n_iterations

        if show_traces:
            all_time_shortest_path = self.run_and_show_traces(single_iteration, once_every = once_every, n_iterations = self.n_iterations, rho_step = rho_step)
        else:
            for i in range(self.n_iterations):
                shortest_path = single_iteration(i, once_every, rho_step)
                if shortest_path[1] < all_time_shortest_path[1]:
                    all_time_shortest_path = shortest_path 

        return all_time_shortest_path


    def pick_move(self, d_level, current):
        """
        Pick the next move of the ant, given the pheromone and heuristic matrices.
        """
        if d_level == 0:
            outbound_pheromones = self.tau_start
            outbound_heuristics = np.ones(self.domain.size)
        else:
            outbound_pheromones = self.tau[d_level-1,current,:]
            outbound_heuristics = self.eta[d_level-2, current, :]

        row = (outbound_pheromones ** self.alpha) * (outbound_heuristics ** self.beta)
        norm_row = (row / row.sum()).flatten()

        if norm_row is np.NaN:
            raise ValueError("The row is NaN")
        return np.random.choice(range(self.domain.size), 1, p = norm_row)[0]


    def generate_ant_path(self, verbose = False):
        # No need to specify the start node, all the ants start from the "start" node
        path = []
        prev = -1
        for d_level, task_id in enumerate(self.tasks):
            current = prev
            move = self.pick_move(d_level, current) if d_level != self.task_graph.n_nodes else np.inf
            path.append((task_id, current, move))
            prev = move

        if verbose:
            print("The path found by the ant is:", path)
        return path


    def path_length(self, path, verbose = False):
        """
        Compute the "length" of the path using the NoC simulator.
        """

        # constuct the mapping form the path
        mapping = {task_id : int(pe) for task_id, pe, _ in path if task_id != "start" and task_id != "end"}

        mapper = mp.Mapper()
        mapper.init(self.task_graph, self.domain)
        mapper.set_mapping(mapping)
        mapper.mapping_to_json(CONFIG_DUMP_DIR + "/dump.json", file_to_append=ARCH_FILE)
        
        if verbose:
            mapper.plot_mapping_gif()

        stub = ss.SimulatorStub()
        result = stub.run_simulation(CONFIG_DUMP_DIR + "/dump.json")
        return result


    def generate_colony_paths(self):
        colony_paths = []
        for _ in range(self.n_ants):
            ant_path = self.generate_ant_path()
            colony_paths.append((ant_path, self.path_length(ant_path)))
        return colony_paths
    
    def evaporate_pheromones(self, step):
        if self.rho is not None:
            self.rho += step
        self.tau = (1 - self.rho) * self.tau

    def update_pheromones(self, colony_paths):
        if self.n_best is None:
            self.n_best = len(colony_paths)
        sorted_paths = sorted(colony_paths, key = lambda x : x[1])
        best_paths = sorted_paths[:self.n_best]
        for path, path_length in best_paths:
            for d_level, path_node in enumerate(path):
                if path_node[1] == -1:
                    self.tau_start[path_node[2]] += 1 / path_length
                elif d_level-1 < self.tau.shape[0]:
                    self.tau[d_level-1, path_node[1], path_node[2]] += 1 / path_length

    def update_heuristics(self):
        """
        Introduce stochasticity in the heuristic matrix.
        """
        self.eta = np.random.rand(self.task_graph.n_nodes-1, self.domain.size, self.domain.size)
    