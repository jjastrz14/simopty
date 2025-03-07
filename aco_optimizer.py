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

import optimizers as op
import models
import partitioner
from dirs import *
      

def single_aco(params, grid, task_graph, file_name_json = "/test.json", save_more_json = False, seed = None):

    print("Starting Single ACO...")
    opt = op.AntColony(params, grid, task_graph, seed = seed)
    
    if save_more_json:
        shortest = opt.run_with_saves(once_every=1, show_traces= False) #run with init, middle and best saves
    else:
        shortest = opt.run(once_every=1, show_traces= False)
        
    print(shortest[0])
    print(opt.path_length(shortest[0], verbose = True))
    opt.save_path_json(shortest[0], SAVE_DATA_DIR + file_name_json)
    
    print("After Single ACO...")
    
    #Load the statistics and plot the results
    #stats = np.load("visual/conv_stats.npy", allow_pickle=True).item()
    #print(stats)

def parallel_ACO(params, grid, task_graph, n_procesors = 10, file_name_json = "/test.json", save_more_json = False, seed = None):
    
    n_procs = n_procesors 
    
    opt = op.ParallelAntColony(n_procs, params, grid, task_graph, seed = seed)
    
    print("Starting Parallel ACO...")
    
    if save_more_json:
        shortest = opt.run_with_saves(once_every=1, show_traces= False) #run with init, middle and best saves
    else:
        shortest = opt.run(once_every=1, show_traces= False)
        
    print(f"Print shortest[0]: {shortest[0]}")
    print(f"Best path of this run: {shortest[1]}")
    print(opt.path_length(shortest[1], verbose = True))
    opt.save_path_json(shortest[1], SAVE_DATA_DIR + file_name_json)
    
    print("After Parallel ACO...")


def GA_algo_run(grid, task_graph, file_name_json = "/test.json", save_more_json = False):
    
    params = op.GAParameters(
        sol_per_pop = 40,
        n_parents_mating=30,
        n_generations = 100,
        mutation_probability = .5,
        crossover_probability = .9,
    )
    print("Starting GA...")
    opt = op.GeneticAlgorithm(params, grid, task_graph)
    shortest = opt.run()
    opt.ga_instance.plot_fitness()
    print(shortest[0], 1/shortest[1])
    opt.save_path_json(shortest[0], SAVE_DATA_DIR + file_name_json)
    print("After GA...")
    

if __name__ == "__main__":
    
    params = op.ACOParameters(
        n_ants = 10,
        rho = 0.05,
        n_best = 20,
        n_iterations = 10,
        alpha = 1.,
        beta = 1.2,
    )
    
    #model = models.test_model((28, 28, 1), verbose = True)
    #model = models.test_model((10, 10, 3), verbose = True)
    model = models.small_test_model((10, 10, 3), verbose = True)
    
    grid, task_graph = partitioner.partitioner(model, n_pe = 4, source = 0, drain = 3)
    
    single_aco(params, grid, task_graph, file_name_json = "/test.json", 
               save_more_json = False, seed = None)
    
    


