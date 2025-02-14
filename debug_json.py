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

if __name__ == "__main__":

    # Create a SimulatorStub object
    stub = ss.SimulatorStub(EX_DIR)

    # Run the simulation
    processors = list(range(6))
    # #config_files = [os.path.join(RUN_FILES_DIR, f) for f in os.listdir(RUN_FILES_DIR) if f.endswith('.json')]
    # results, logger = stub.run_simulations_in_parallel(config_files=config_files, processors=processors, verbose=True)
    # results, logger = stub.run_simulation("config_files/dumps/dump.json", verbose = True)

    # #path to save the data
    path_data = SAVE_DATA_DIR + "/test.json"
    #path_data = "config_files/test_recon.json"
    #path_data = "config_files/runs/test_run.json"
    results, logger = stub.run_simulation(path_data, verbose = False)
    print(results)
        
    # for event in logger.events:
    #     print(event)
    #     print(f"Event ID: {event.id}, Type: {event.type}, Cycle: {event.cycle}, Additional info: {event.additional_info}," 
    #           f"Info: {event.info}")
    #     if event.type == nocsim.EventType.START_COMPUTATION:
    #         print(f"Node ID: {event.info.node}")
    #     elif event.type == nocsim.EventType.OUT_TRAFFIC:
    #         print(f"History: {event.info.history}")
    #     elif event.type == nocsim.EventType.START_RECONFIGURATION:
    #         print(f"Node ID: {event.info.node}")
    # print("-------------------")

    plotter_timeline = NoCTimelinePlotter()
    plotter_3d_animation = NoCPlotter()
    timeline_path = "visual/test_model_13_Feb_6.png"
    gif_path = "visual/test_model_13_Feb_6.gif"

    print("Plotting timeline...")
    # Generate 2D timeline
    plotter_timeline.setup_timeline(logger, path_data)
    plotter_timeline.plot_timeline(timeline_path)
    
    # fps = 200
    # start_time = time.time()
    # print("Plotting 3D animation...")
    # plotter_3d_animation.plot(logger, fps, path_data, gif_path, verbose = False)  # Original 3D plot
    # end_time = time.time()
    # print(f"3D animation plotting took {end_time - start_time:.2f} seconds")


