'''
==================================================
File: main.py
Project: simopty
File Created: Sunday, 8th December 2024
Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''


"""
The main.py module contains the main function of the program.
"""

import os
import graph as dg
import domain as dm
import mapper as mp
import simulator_stub as ss
from dirs import *
import optimizers as op


if __name__ == "__main__":
    
    # Create a DepGraph object
    dep_graph = dg.DepGraph()

    # Define the structure of the dependency graph
    dep_graph.add_task_fully_qualified(id=3, type = "COMP_OP", layer_id = 0,  ct_required=40, dep = [0])
    dep_graph.add_task_fully_qualified(id=4, type = "COMP_OP", layer_id = 0,  ct_required=30, dep = [1])
    dep_graph.add_task_fully_qualified(id=5, type = "COMP_OP", layer_id = 0,  ct_required=20, dep = [2])
    dep_graph.add_dependency_fully_qualified("start", 3, id = 0, type = "WRITE", size = 8, pt_required= 8, cl = 0, dep= [-1])
    dep_graph.add_dependency_fully_qualified("start", 4, id = 1, type = "WRITE", size = 6, pt_required= 6, cl = 0, dep= [-1])
    dep_graph.add_dependency_fully_qualified("start", 5, id = 2, type = "WRITE", size = 4, pt_required= 4, cl = 0, dep= [-1])
    dep_graph.add_task_fully_qualified(id = 11, type = "COMP_OP", layer_id = 1, ct_required = 10, dep = [6, 7, 9])
    dep_graph.add_task_fully_qualified(id = 12, type = "COMP_OP", layer_id = 1, ct_required = 10, dep = [8, 10])
    dep_graph.add_dependency_fully_qualified(3, 11, id = 6, type = "WRITE_REQ",  size = 4, pt_required = 4, cl = 0, dep = [3])
    dep_graph.add_dependency_fully_qualified(4, 11, id = 7, type = "WRITE_REQ",  size = 1, pt_required = 1, cl = 0, dep = [4])
    dep_graph.add_dependency_fully_qualified(5, 11, id = 9, type = "WRITE_REQ",  size = 1, pt_required = 1, cl = 0, dep = [5])
    dep_graph.add_dependency_fully_qualified(3, 12, id = 8, type = "WRITE_REQ",  size = 2, pt_required = 2, cl = 0, dep = [4])
    dep_graph.add_dependency_fully_qualified(5, 12, id = 10, type = "WRITE_REQ",  size = 1, pt_required = 1, cl = 0, dep = [5])
    dep_graph.add_task_fully_qualified(id = 15, type = "COMP_OP", layer_id = 2,  ct_required = 10, dep = [13, 14])
    dep_graph.add_dependency_fully_qualified(11, 15, id = 13, type = "WRITE_REQ",  size = 1, pt_required = 1, cl = 0, dep = [11])
    dep_graph.add_dependency_fully_qualified(12, 15, id = 14, type = "WRITE_REQ",  size = 1, pt_required = 1, cl = 0, dep = [12])
    dep_graph.add_dependency_fully_qualified(15, "end", id = 16, type = "WRITE",  size = 1, pt_required = 1, cl = 0, dep = [15])
    # dep_graph.plot_graph()
    # Create a Grid object
    grid = dm.Grid()
    grid.init(3, 2, dm.Topology.TORUS)

    # Create a Mapper object
    mapper = mp.Mapper()
    mapper.init(dep_graph, grid)


    # Ant's job: decide the mapping
    # mapping = {3 : 0, 4 : 1, 5 : 2, 11 : 3, 12 : 4, 15 : 7}
    # mapping = {3 : 0, 4 : 1, 5 : 2, 11 : 6, 12 : 4, 15 : 7}
    # mapping = {3 : 0, 4 : 3, 5 : 2, 11 : 6, 12 : 4, 15 : 7}
    # mapping = {3 : 3, 4 : 3, 5 : 2, 11 : 6, 12 : 4, 15 : 7}


    # mapper.set_mapping(mapping)
    # mapper.plot_mapping_gif()
    # # Create the configuration file from the arch and the structure
    # mapper.mapping_to_json(CONFIG_DUMP_DIR + "/dump1.json", file_to_append=ARCH_FILE)

    # Create a SimulatorStub object
    stub = ss.SimulatorStub(EX_DIR)

    # Run the simulation
    # processors = list(range(6))
    # config_files = [os.path.join(RUN_FILES_DIR, f) for f in os.listdir(RUN_FILES_DIR) if f.endswith('.json')]
    # results = stub.run_simulations_in_parallel(config_files=config_files, processors=processors, verbose=True)
    # results = stub.run_simulation(CONFIG_DUMP_DIR + "/dump1.json", verbose = True)
    # print(results)

    # Define a Optimization object
    params = op.OptimizationParameters(
        n_ants = 20,
        rho = 0.05,
        n_best = 10,
        n_iterations = 150,
        alpha = 1.,
        beta = 1.2,
    )
    #something here

    opt = op.AntColony(params, grid, dep_graph)

    shortest = opt.run(once_every=1, show_traces= True)
    print(shortest[0])
    print(opt.path_length(shortest[0], verbose = True))


