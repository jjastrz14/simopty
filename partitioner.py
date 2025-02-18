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

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations
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
from dataclasses import dataclass
from utils.partitioner_utils import *


def test_conv(input_shape, verbose = False):
    
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(4, kernel_size=(3, 3), data_format="channels_last", activation=None) (inputs)
    model = keras.Model(inputs=inputs, outputs=x)
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    if verbose:
        summary = model.summary()
        print(f'shape of model: {x.shape}')
    return model


if __name__ == "__main__":
    
    
    model = test_conv((10, 10, 3), verbose = True)

    n_proc = 2
    grid = dm.Grid()
    grid.init(n_proc, 2, dm.Topology.TORUS)

    drain = 3
    source = 0 #coś tu jest nie tak kiedy drain and source are the same
    #should be in the restart
    
    if ((n_proc * n_proc) - 1) < drain:
        raise ValueError(f"The number of processors: {n_proc * n_proc - 1} must be greater than the drain node: {drain}")
    
    task_graph = model_to_graph(model, source = source, 
                                drain = drain, 
                                grouping = False, 
                                verbose=True)
    
    plot_graph(task_graph, file_path = "../task_graph.png")








