'''
==================================================
File: parallel_aco_utils.py
Project: utils
File Created: Thursday, 26th December 2024
Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''

"""
The parallel_aco_utils.py module contains the classes and functions used to implement the parallel version of the Ant Colony Optimization algorithm.
"""

import ctypes as c
import numpy as np
import simulator_stub as ss
import mapper as mp
from dirs import *

# dictionary to store the shared variables (pheromone and heuristic matrices)
vardict = {}

class Ant:

    def __init__(self, id, graph, domain, tasks, alpha, beta):
        self.id = id
        self.graph = graph
        self.domain = domain
        self.tasks = tasks
        self.alpha = alpha
        self.beta = beta

        self.current_path = None


    def pick_move(self, d_level, current):
        """
        Pick the next move of the ant, given the pheromone and heuristic matrices.
        """
        global vardict

        tau_start = np.frombuffer(vardict["tau_start"].get_obj())
        tau = np.frombuffer(vardict["tau"].get_obj()).reshape(vardict["tau.size"])
        eta = np.frombuffer(vardict["eta"].get_obj()).reshape(vardict["eta.size"])

        if d_level == 0:
            outbound_pheromones = tau_start
            outbound_heuristics = np.ones(self.domain.size)
        else:
            outbound_pheromones = tau[d_level-1,current,:]
            outbound_heuristics = eta[d_level-2, current, :]
        row = (outbound_pheromones ** self.alpha) * (outbound_heuristics ** self.beta)
        norm_row = (row / row.sum()).flatten()
        if norm_row is np.NaN:
            raise ValueError("The row is NaN")
        return np.random.choice(range(self.domain.size), 1, p = norm_row)[0]
    
    def path_lenght(self, graph, domain, path):
        """
        Compute the lenght of the path.
        """

        # constuct the mapping form the path
        mapping = {task_id : int(pe) for task_id, pe, _ in path if task_id != "start" and task_id != "end"}

        mapper = mp.Mapper()
        mapper.init(graph, domain)
        mapper.set_mapping(mapping)
        mapper.mapping_to_json(CONFIG_DUMP_DIR + "/dump{}.json".format(self.id), file_to_append=ARCH_FILE)
        

        stub = ss.SimulatorStub()
        result = stub.run_simulation(CONFIG_DUMP_DIR + "/dump{}.json".format(self.id))
        return result
        
    def walk(self):
        """
        Perform a walk of the ant on the grid.
        """
        path = []
        prev = -1
        for d_level, task_id in enumerate(self.tasks):
            current = prev
            move = self.pick_move(d_level, current) if d_level != self.graph.n_nodes else np.inf
            path.append((task_id, current, move))
            prev = move
        
        self.current_path = (self.id, path, self.path_lenght(self.graph, self.domain, path))
        return self.current_path
    
    @staticmethod
    def update_pheromones(path):
        """
        Update the pheromone matrix in the shared memory.
        """

        tau_start = np.frombuffer(vardict["tau_start"].get_obj())
        tau = np.frombuffer(vardict["tau"].get_obj()).reshape(vardict["tau.size"])

        with vardict["tau_start"].get_lock():
            path_node = path[1][0]
            assert path_node[1] == -1
            tau_start[path_node[2]] += 1/path[2]

        with vardict["tau"].get_lock():
            for d_level, path_node in enumerate(path[1][1:]):
                if d_level-1 < vardict["tau.size"][0]:
                    tau[d_level-1, path_node[1], path_node[2]] += 1/path[2]

    @staticmethod
    def update_heuristics():
        """
        Update the heuristic matrix in the shared memory.
        """

        pass

    
        