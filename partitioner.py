'''
==================================================
File: partitioner.py
Project: simopty
File Created: Tuesday, 31st December 2024
Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''

"""
The partitioner.py module contains the classes and functions needed to create a partition/task graph out of a 
given DNN model. The chosen library for this purpose is TensorFlow.

"""


from graph import model_to_graph
import domain as dm
from dirs import *
from utils.partitioner_utils import *
import models 


def partitioner(model, n_pe = 2, source = 0, drain = 3):
    
    grid = dm.Grid()
    grid.init(n_pe, 2, dm.Topology.TORUS)

    #source and drain should be defined in restart!
    #coś tu jest nie tak kiedy drain and source are the same
    #should be in the restart
    
    if ((n_pe * n_pe) - 1) < drain:
        raise ValueError(f"The number of processors: {n_pe * n_pe - 1} must be greater than the drain node: {drain}")
    
    task_graph = model_to_graph(model, source = source, 
                                drain = drain, 
                                grouping = False,
                                namefile_task_graph = "visual/task_graph_conv.png",
                                verbose=True)
    #plot_graph(task_graph, file_path = "../task_graph.png")
    
    return grid, task_graph

if __name__ == "__main__":
    
    model = models.test_conv((10, 10, 3), verbose = True)
    #model = models.test_model((10, 10, 3), verbose = True)
    analyze_ops(model,incl_info = False)
    task_graph = partitioner(model)









