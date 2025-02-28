'''
==================================================
File: main.py
Project: simopty
File Created: Sunday, 8th December 2024
Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it) Jakub Jastrzebski (jakubandrzej.jastrzebski@polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''

"""
The main.py module contains the main function of the program.
"""

import optimizers as op
import models
import partitioner
import aco_optimizer
import visualiser

if __name__ == "__main__":
    
    model = models.test_conv((10, 10, 3), verbose = True)
    
    grid, task_graph = partitioner.partitioner(model, n_pe = 2, source = 0, drain = 3)

    params = op.ACOParameters(
        n_ants = 10,
        rho = 0.05,
        n_best = 20,
        n_iterations = 10,
        alpha = 1.,
        beta = 1.2,
    )
    
    n_procesors = 10
    
    file_name_json = "/test.json"
    
    #simulator
    aco_optimizer.single_aco(params, grid, task_graph, file_name_json = file_name_json, save_more_json = False)
    #aco_optimizer.parallel_ACO(params, grid, task_graph, n_procesors = n_procesors, file_name_json = file_name_json, save_more_json = False)
    
    #visualisations
    path_timeline = "visual/timeline.png"
    path_gif = "visual/test.gif"
    path_animation = "visual/combined_animation.gif"
    
    visualiser.plot_timeline(file_name_json, path_timeline, verbose = False)
    #visualiser.plot_3d_animaiton(file_name_json, fps = 2, gif_path = path_gif)
    #visualiser.generate_animation_timeline_3d_plot(file_name_json, fps = 2, animation_path = path_animation)
    
    print("Done!")
    


