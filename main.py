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
from graph import model_to_graph
import domain as dm
import mapper as mp
import simulator_stub as ss
from dirs import *
import optimizers as op
from utils.plotting_utils import *
from utils.ga_utils import *
from utils.partitioner_utils import *
from utils.ani_utils import *
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.utils import plot_model
import time
import nocsim
import models
import partitioner

if __name__ == "__main__":
    
    
    # Define a Optimization object
    # params = op.ACOParameters(
    #     n_ants = 20,
    #     rho = 0.05,
    #     n_best = 10,
    #     n_iterations = 150,
    #     alpha = 1.,
    #     beta = 1.2,
    # )
    # opt = op.AntColony(params, grid, dep_graph)
    # shortest = opt.run(once_every=1, show_traces= True)
    # print(shortest[0])
    # print(opt.path_length(shortest[0], verbose = True))


    # params = op.GAParameters(
    #     sol_per_pop = 40,
    #     n_parents_mating=30,
    #     n_generations = 100,
    #     mutation_probability = .5,
    #     crossover_probability = .9,
    # )
    # opt = op.GeneticAlgorithm(params, grid, dep_graph)
    # shortest = opt.run()
    # opt.ga_instance.plot_fitness()
    # print(shortest[0], 1/shortest[1])

    #model = models.test_model((28, 28, 1), verbose = True)
    model = models.test_model((10, 10, 3), verbose = True)
    #model = models.small_test_model((10, 10, 3), verbose = True)
    
    grid, task_graph = partitioner.partitioner(model, n_pe = 4, source = 0, drain = 3)

    params = op.ACOParameters(
        n_ants = 8,
        rho = 0.05,
        n_best = 20,
        n_iterations = 1,
        alpha = 1.,
        beta = 1.2,
    )
    
    print("Starting ACO...")
    
    # opt = op.AntColony(params, grid, task_graph)
    # shortest = opt.run(once_every=1, show_traces= False)
    # print(shortest[0])
    # print(opt.path_length(shortest[0], verbose = True))
    # opt.save_path_json(shortest[0], SAVE_DATA_DIR + "/test_big.json")

    n_procs = 8 #n_processors shouldn't be greater than n_ants
    #shortest = opt.run_with_saves(once_every=1, show_traces= False) #run with init, middle and best saves
    opt = op.ParallelAntColony(n_procs, params, grid, task_graph)
    shortest = opt.run(once_every=1, show_traces= False)
    print(f"Print shortest[0]: {shortest[0]}")
    print(f"Best path of this run: {shortest[1]}")
    print(opt.path_length(shortest[1], verbose = True))
    opt.save_path_json(shortest[1], SAVE_DATA_DIR + "/test_big.json")
    print("After ACO...")
            
    # # # Load the statistics and plot the results
    # # # stats = np.load("visual/statistics.npy", allow_pickle=True).item()
    # # # print(stats)


