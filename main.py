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
import time

def test_model(input_shape):
    
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(3, kernel_size=(3, 3), activation='linear') (inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(6, kernel_size=(3, 3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dense(24, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=x)
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    summary = model.summary()
    return model


def load_model(model_name):
    available_models = ["ResNet50", "MobileNetV2", "MobileNet", "ResNet18"]
    if model_name not in available_models:
        raise ValueError(f"Model not available. Please choose from the following: {', '.join(available_models)}")
    
    # Load the model
    model = keras.applications.__getattribute__(model_name)(weights='imagenet')
    return model

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

    model = test_model((28, 28, 1))
    # # # # # model = load_model("ResNet50")
    # # # # # model = load_model("MobileNet")
    # # # # # model = load_model("MobileNetV2")
    # # model = test_model((28, 28, 1))
    # # # # # model = load_model("ResNet50")
    # # # # # model = load_model("MobileNet")
    # # # # # model = load_model("MobileNetV2")

    # # model.summary()
    # # plot_model(model, to_file="visual/model.png", show_shapes=True)
    # # #analyze_ops(model, False)

    # # # # # print(split_spatial_dims(model.layers[2], 2))
    
    task_graph = model_to_graph(model, verbose=True)
    #plot_graph(task_graph)

    grid = dm.Grid()
    grid.init(6, 2, dm.Topology.TORUS)

    params = op.ACOParameters(
        n_ants = 100,
        rho = 0.05,
        n_best = 20,
        n_iterations = 2,
        alpha = 1.,
        beta = 1.2,
    )
    
    n_procs = 100 #n_processors shouldn't be greater than n_ants

    opt = op.ParallelAntColony(n_procs, params, grid, task_graph)
    # # # #opt = op.AntColony(params, grid, task_graph)

    # # # #shortest = opt.run_with_saves(once_every=1, show_traces= False) #run with init, middle and best saves
    shortest = opt.run(once_every=1, show_traces= False)
    print(shortest[1])
    print(opt.path_length(shortest[1], verbose = True))
    opt.save_path_json(shortest[1], SAVE_DATA_DIR + "/all_time_shortest_path.json")
            
    # # # Load the statistics and plot the results
    # # # stats = np.load("visual/statistics.npy", allow_pickle=True).item()
    # # # print(stats)

    # Create a SimulatorStub object
    stub = ss.SimulatorStub(EX_DIR)

    # Run the simulation
    processors = list(range(6))
    # #config_files = [os.path.join(RUN_FILES_DIR, f) for f in os.listdir(RUN_FILES_DIR) if f.endswith('.json')]
    # results, logger = stub.run_simulations_in_parallel(config_files=config_files, processors=processors, verbose=True)
    # results, logger = stub.run_simulation("config_files/dumps/dump.json", verbose = True)

    # #path to save the data
    path_data = SAVE_DATA_DIR + "/all_time_shortest_path.json"
    #path_data = "config_files/test_recon.json"
    #path_data = "config_files/runs/test_run.json"
    results, logger = stub.run_simulation(path_data, verbose = False)
    print(results)
    
    #Initialize plotter with timeline support
    plotter_3d_animation = NoCPlotter()
    plotter_timeline = NoCTimelinePlotter()
    
    fps = 100
    # # #paths
    #gif_path = "visual/test.gif"
    timeline_path = "visual/test.png"
    
    # start_time = time.time()
    # print("Plotting 3D animation...")
    # plotter_3d_animation.plot(logger, fps, path_data, gif_path, verbose = False)  # Original 3D plot
    # end_time = time.time()
    # print(f"3D animation plotting took {end_time - start_time:.2f} seconds")

    print("Plotting timeline...")
    # Generate 2D timeline
    plotter_timeline.setup_timeline(logger, path_data)
    plotter_timeline.plot_timeline(timeline_path)
    #plotter_timeline._print_node_events()
        
    # for event in logger.events:
    #     print(event)
    #     print(f"Event ID: {event.id}, Type: {event.type}, Cycle: {event.cycle}, Additional info: {event.additional_info}," 
    #           f"Info: {event.info}")
    #     # if event.type == nocsim.EventType.START_COMPUTATION:
    #     #     print(f"Node ID: {event.info.node}")
    #     # elif event.type == nocsim.EventType.OUT_TRAFFIC:
    #     #     print(f"History: {event.info.history}")
    #     # elif event.type == nocsim.EventType.START_RECONFIGURATION:
    #     #     print("-------------------")
    #     #     print(f"Found: {event.type}")
    #     #     print("-------------------")
    #     # else:
    #     #     print(f"I don't know how to handle this event: {event.type}")
    # print("-------------------")



