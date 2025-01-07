'''
==================================================
File: partitioner_utils.py
Project: utils
File Created: Tuesday, 31st December 2024
Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
Under the supervision of: Politecnico di Milano
==================================================

Special thanks to: 1. https://github.com/Lyken17/pytorch-OpCounter
                   2. https://github.com/MrYxJ/calculate-flops.pytorch
'''




"""
* = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = 
1. LAYER HOOKS: used to esimate the flops for a certain operation/layer.

Args:
    - input_shape: the shape of the input tensor to the layer
    - output_shape: the shape of the output tensor of the layer
    - layer : the instance of the layer for which we want to estimate the FLOPs
Return:
    - the number of FLOPs for the layer
    - the number of MACS for the layer
* = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = 
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import keras.layers as layers



# Tranformation layers

def _calc_input(layer, input_shape, output_shape):
    return 0,0

def _calc_reshape(layer, input_shape, output_shape):
    return 0,0

def _calc_flatten(layer, input_shape, output_shape):
    return 0,0

def _calc_identity(layer, input_shape, output_shape):
    return 0,0

# Add layers
def _calc_add(layer, input_shape, output_shape):
    return np.prod(output_shape[1:]), 0

# Activation layers

def _calc_linear(layer, input_shape, output_shape):
    return 0,0

def _calc_relu(layer, input_shape, output_shape):
    return np.prod(output_shape[1:]), 0

def _calc_sigmoid(layer, input_shape, output_shape):
    return 4*np.prod(output_shape[1:]), 0

def _calc_tanh(layer, input_shape, output_shape):
    return 6*np.prod(output_shape[1:]), 0

def _calc_elu(layer, input_shape, output_shape):
    return 6*np.prod(output_shape[1:]), 0

def _calc_relu6(layer, input_shape, output_shape):
    return 2*np.prod(output_shape[1:]), 0

def _calc_leaky_relu(layer, input_shape, output_shape):
    return 4*np.prod(output_shape[1:]), 0

def _calc_softmax(layer, input_shape, output_shape):
    return (3*np.prod(output_shape[1:])) + 1, 0


def _calc_activation(layer, inputs_shape: tuple, outputs_shape: tuple):
    '''
    A hook serving as a router to the different activation functions.
    '''
    # Determine the type of activation function form the activation layer
    activation = layer.activation.__name__
    if activation == 'linear':
        return _calc_linear(layer, inputs_shape, outputs_shape)
    elif activation == 'relu':
        return _calc_relu(layer, inputs_shape, outputs_shape)
    elif activation == 'sigmoid':
        return _calc_sigmoid(layer, inputs_shape, outputs_shape)
    elif activation == 'tanh':
        return _calc_tanh(layer, inputs_shape, outputs_shape)
    elif activation == 'elu':
        return _calc_elu(layer, inputs_shape, outputs_shape)
    elif activation == 'relu6':
        return _calc_relu6(layer, inputs_shape, outputs_shape)
    elif activation == 'leakyrelu':
        return _calc_leaky_relu(layer, inputs_shape, outputs_shape)
    elif activation == 'softmax':
        return _calc_softmax(layer, inputs_shape, outputs_shape)

# Zero Padding layers

def _calc_zero_padding(layer, input_shape, output_shape):
    return 0,0

# Fully Connected layers

def _calc_fc(layer, input_shape, output_shape):
    bias_flops = np.prod(output_shape[1:]) if layer.use_bias else 0
    MACs  = np.prod(input_shape[1:]) * np.prod(output_shape[1:])
    FLOPs = np.prod(output_shape[1:]) * (2 * np.prod(input_shape[1:]) - 1) + bias_flops

    return FLOPs, MACs

# Pooling layers

def _calc_pool(layer, input_shape, output_shape):
    if isinstance(layer, layers.GlobalAveragePooling2D) or isinstance(layer, layers.GlobalAveragePooling1D):
        return np.prod(output_shape[1:]), 0
    k_size = np.prod(layer.pool_size)
    FLOPs = k_size - 1
    FLOPs *= np.prod(output_shape[1:])
    return FLOPs, 0


# Batch Normalization layers

def _calc_batch_norm(layer, input_shape, output_shape):
    norm_ops = 2 * output_shape[-1]
    norm_ops += 2 * np.prod(output_shape[1:])
    scale_ops = np.prod(output_shape[1:]) * 2 if layer.scale else 0
    bn_flops = norm_ops + scale_ops
    return bn_flops, 0

# Convolutional layers

def _calc_conv(layer, input_shape, output_shape):
    out_dims = output_shape[1:-1]
    out_channels = output_shape[-1]
    kernel_dims = layer.kernel_size
    in_channels = input_shape[-1]
    
    
    MACs = np.prod(kernel_dims) * np.prod(out_dims) * in_channels * out_channels
    FLOPs = 2 * np.prod(kernel_dims) * np.prod(out_dims) * in_channels * out_channels
    # Each output element gets a bias addition
    bias_flops = np.prod(output_shape[1:]) if layer.use_bias else 0

    MACs += bias_flops//2
    FLOPs += bias_flops

    return FLOPs, MACs

def _calc_depthwise_conv(layer, input_shape, output_shape):
    in_dims = input_shape[1:-1]
    in_channels = input_shape[-1]
    kernel_dims = layer.kernel_size
    out_dims = output_shape[1:-1]
    depth_multiplier = layer.depth_multiplier

    # Compute the number of operations required to covolve each channel with individual depthwise kernels, having 
    # depth_multiplier channels in the output
    MACs = np.prod(kernel_dims) * np.prod(out_dims) * in_channels
    FLOPs = 2 * np.prod(kernel_dims) * np.prod(out_dims) * in_channels
    # Concatenate the convoled outputs along the channel axis
    deph_flops = np.prod(out_dims) * (depth_multiplier - 1)
    # Add bias contributions
    bias_flops = np.prod(output_shape[1:]) if layer.use_bias else 0

    MACs += deph_flops//2 + bias_flops//2
    FLOPs += deph_flops + bias_flops

    return FLOPs, MACs

def _calc_transpose_conv(layer, input_shape, output_shape):
    # Numer of flops to determine the paddinng
    padding_flops = len(layer.kernel_size) * 8

    # Can also use layer.kernel_size, layer.filters and layer.input_shape to calculate FLOPs
    MACs, FLOPs = _calc_conv(layer, input_shape, output_shape)

    MACs += padding_flops//2
    FLOPs += padding_flops
    
    return FLOPs, MACs

# Dropout layers
def _calc_dropout(layer, input_shape, output_shape):
    return np.prod(input_shape[1:]), 0


register_hooks = {
    layers.InputLayer: _calc_input,
    layers.Reshape: _calc_reshape,
    layers.Flatten: _calc_flatten,
    layers.Add: _calc_add,
    layers.ZeroPadding1D: _calc_zero_padding,
    layers.ZeroPadding2D: _calc_zero_padding,
    layers.Dense: _calc_fc,
    layers.MaxPooling1D: _calc_pool,
    layers.MaxPooling2D: _calc_pool,
    layers.AveragePooling1D: _calc_pool,
    layers.AveragePooling2D: _calc_pool,
    layers.GlobalAveragePooling2D: _calc_pool,
    layers.GlobalAveragePooling1D: _calc_pool,
    layers.GlobalMaxPooling2D: _calc_pool,
    layers.GlobalMaxPooling1D: _calc_pool,
    layers.BatchNormalization: _calc_batch_norm,
    layers.Conv1D: _calc_conv,
    layers.Conv2D: _calc_conv,
    layers.DepthwiseConv2D: _calc_depthwise_conv,
    layers.DepthwiseConv1D: _calc_depthwise_conv,
    layers.Conv1DTranspose: _calc_transpose_conv,
    layers.Conv2DTranspose: _calc_transpose_conv,
    layers.Dropout: _calc_dropout,
    layers.Identity: _calc_identity,
    layers.ReLU: _calc_relu,
    layers.ELU: _calc_elu,
    layers.Activation: _calc_activation,
    'linear': _calc_linear,
    'relu': _calc_relu,
    'sigmoid': _calc_sigmoid,
    'tanh': _calc_tanh,
    'elu': _calc_elu,
    'relu6': _calc_relu6,
    'leakyrelu': _calc_leaky_relu,
    'softmax': _calc_softmax
}

"""
* = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = *
2. PARTITIONING STRATEGIES: used to partition the workload of a layer among the different PEs in the NoC.

We define 3 different ways of executing the partitioning of the model at layer-level-granularity:

- SPATIALLY: the partitioning is done by splitting the spatial dimensions of the layer among the different partitions:
            This means that each partition will own a subset of the spatial dimensions of the layer, together with the 
            weights and biases of all the layer's kernels. This approach is particulary useful when the input data is
            too large to fit in the memory of a single PE, while there are still a small number of input and output channels.

( The next two approaches are only valid for layers admitting a 4D input tensor, such as Conv2D, Conv2DTranspose, BatchNormalization, etc. ) 
       
- BY INPUT_CHANNELS: the partitioning is done by splitting the input channels of the layer among the different partitions:
                     This means that each partition will own a subset of the input channels of the layer, together with the 
                     corresponding weights and biases of all the layer's kernels for the chosen input channels. This approach is 
                     particularly useful when the input data is small enough to fit in the memory of a single PE, while there are
                     still a large number of input channels and small amount of output channels. It also induces the need for a 
                     communication phase between the partitions, in order to exchange output data

- BY OUTPUT_CHANNELS: the partitioning is done by splitting the output channels of the layer among the different partitions:
                     This means that each partition will own a subset of the output channels of the layer, together with the
                     corresponding weights and biases of the corresponding chosen kernels. This approach is particularly useful when
                     the input data is small enough to fit in the memory of a single PE, while there are still a large number of output
                     channels and small amount of input channels.

Since most of the time these approaches will be used in combination, we also define a fourth approach, which is a combination of the
previous three, using some tuning parameters:
    - in_sp: the percentage of the spatial dimensions of the layer that will be assigned a single partition
    - out_ch: the percentage of the output channels of the layer that will be assigned a single partition
    - in_ch: the percentage of the input channels of the layer that will be assigned a single partition

Using these parameters, we can define a custom partitioning strategy. In particular, choosing:
    - in_sp = x, out_ch = 0, in_ch = 0: will result in a pure spatial partitioning, where x% of the input is assgned to a single partition
    - in_sp = 0, out_ch = x, in_ch = 0: will result in a pure output channel partitioning, where x% of the output channels are assigned to a single partition
    - in_sp = 0, out_ch = 0, in_ch = x: will result in a pure input channel partitioning, where x% of the input channels are assigned to a single partition

* = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = *
"""

from dataclasses import dataclass
from typing import List, Union
from copy import deepcopy
import string
import pydot
import matplotlib.pyplot as plt
import graph as dg
import math

@dataclass
class PE:
    """
    A simple dataclass to represent a Processing Element (PE) in the NoC. This will be just used to keep track
    of the PE's available resources (memory)
    """
    # The space used to store in memory a single number (input/weights/output) (in bytes)
    single_num: int = 1
    # The size of the PE's memory (in bytes)
    mem_size: int = 512000
    # The amount of memory used by the PE (in bytes)
    mem_used: int = 0

    def __init__(self, mem_size: int = None):
        self.mem_size = mem_size
        self.mem_used = 0

# A function to construct the id to assign to each partition. The general rule for construction of the ID of 
# a partition is the following:
# for each type of partitioning, a unique number is assigned to the partition: since multiple partitions may be applied sequentially to the same layer, and since the order of
# in which the partitioning is applied is not important (the results produced are the same), we build the ID by concatenating the :
# - the layer number in the model,
# - the local ID of the spatial partitioning (if no partitioning is applied, the ID is x),
# - the local ID of the output channel partitioning (if no partitioning is applied, the ID is x),
# - the local ID of the input channel partitioning (if no partitioning is applied, the ID is x).
# As the order in which we perform the partitions is not important, we chose to always perform multiple partitions in the order: spatial-> ouput -> input.
# The "local" id of the partition is computed, based on the previous partition's local id


# A function to build the fully qualified ID of the partition
def _build_id(layer_name, id, partition_type, partition_local_id):
    id_list = id.split('-')
    # check that the layer name does not contain the "x" of "-" character
    if "-" in layer_name:
        raise ValueError("The layer name cannot contain the character '-'")

    if id_list[0] == 'x':
        id_list[0] = str(layer_name)
    else:
        assert id_list[0] == layer_name, "The layer name is different from the one in the ID"

    if partition_type == 'spatial':
        id_list[1] = str(partition_local_id)
    elif partition_type == 'output':
        id_list[2] = str(partition_local_id)
    elif partition_type == 'input':
        id_list[3] = str(partition_local_id)

    return '-'.join(id_list)   

# A datastructure to hold the information for the partition
@dataclass
class PartitionInfo:

    def __init__(self, layer, **kwargs):
        self.layer = layer
        self.id = 'x-x-x-x'
        self.task_id = None
        self.layer_id = None
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.MACs = 0
        self.FLOPs = 0

    def set_id(self, partition_type, partition_local_id, previous_id = None):
        if previous_id is not None:
            self.id = previous_id
        self.id = _build_id(self.layer.name, self.id, partition_type, partition_local_id)

    def set_split_factor(self, type, split_factor):
        if type == 'spatial':
            self.spatial_sp_fact = split_factor
        elif type == 'output':
            self.output_sp_fact = split_factor
        elif type == 'input':
            self.input_sp_fact = split_factor
        else:
            return ValueError("Invalid partition type")
    # Partition id
    id: str
    task_id: Union[None, int]

    # Layer name
    layer: keras.layers.Layer
    layer_id: Union[None, int]

    # Information on the partition
    # input bounds
    in_bounds: Union[None, List[tuple]] = None
    # output bounds
    out_bounds: Union[None, List[tuple]] = None
    # input channels included in the partiion
    in_ch: Union[None, List[tuple]] = None
    # output channels included in the partiton
    out_ch: Union[None, List[tuple]] = None
    # weights shape for the partition
    weights_shape: Union[None, List[tuple]] = None



def _split_spatial_dims(layer, split_factor, partitions: Union[None, List[PartitionInfo]] = None):
    '''
    A function to compute the spatial partitioning of the input and output tensors of a layer.
    If a partitions list is given, for each object in the list, the partion strategy is applied to the object and a new list
    of partitions is created.

    Args:
    - layer : the layer to be partitioned
    - split_factor : the number of partitions to create -> the split factor is used to compute the number of partitions as a power of 2
    - partitions : the partition list on which to apply the partitioning

    Returns:
    - a new PartitionInfo list
    '''
    

    def _create_partitions(input_dims, output_dims, weights_dims, partition = None):

        new_partitions = []
        original_input_dims = deepcopy(input_dims)
        original_output_dims = deepcopy(output_dims)
        input_dims = deepcopy(input_dims)
        output_dims = deepcopy(output_dims)

        s_x = (split_factor // 2 + split_factor % 2) * 2
        s_y = (split_factor // 2) * 2

        # Check the layer input shape
        if split_factor > 0:
            if len(input_dims) > 2:
                overlap = layer.kernel_size[0] //2  if isinstance(layer, (layers.Conv2D, layers.DepthwiseConv2D)) else 0 # at this stage don't consider the stride
            
                input_dims[0] = (input_dims[0] + s_x - 1) // s_x if s_x > 1 else input_dims[0]
                input_dims[1] = (input_dims[1] + s_y - 1) // s_y if s_y > 1 else input_dims[1]
                output_dims[0] = (output_dims[0] + s_x - 1) // s_x if s_x > 1 else output_dims[0]
                output_dims[1] = (output_dims[1] + s_y - 1) // s_y if s_y > 1 else output_dims[1]

                # Add the overlap to the partition size 
                for dim in range(len(input_dims)-1):
                    input_dims[dim] += overlap*2
                    input_dims[dim] = min(input_dims[dim], original_input_dims[dim])
            
            else:
                # if the layer is Dense, we split the partition by considering only ouput neurons:
                # each partition will have the same number of input neurons ( corresponding to the number of input neurons of the layer)
                # and a subset of the output neurons
                overlap = layer.kernel_size[0] //2  if layer in [layers.Conv1D, layers.DepthwiseConv1D] else 0
                index = 0

                input_dims[index] = (input_dims[index]+ s_x - 1) // s_x if s_x > 1 else input_dims[index]
                output_dims[index] = (output_dims[index] + s_x - 1) // s_x if s_x > 1 else output_dims[index]
                
                input_dims[index] += overlap*2
                input_dims[index] = min(input_dims[index], original_input_dims[index])


        # Append the partitions to the new list:
        granularity = 2**split_factor
        for _ in range(granularity):
            in_x_0 = (_ % s_x) * ((original_input_dims[0] + s_x - 1)// s_x) - overlap if s_x > 1 else 0
            out_x_0 = (_ % s_x) * output_dims[0] if s_x > 1 else 0
            if isinstance(layer, layers.Dense) or (isinstance(layer, layers.Activation) and layer.activation.__name__ == 'softmax'):
                in_bounds = [(0,), (original_input_dims[0],)]
            else:
                in_bounds = [(max(in_x_0,0),), (min(in_x_0 + input_dims[0], original_input_dims[0]),)]
            out_bounds = [(max(out_x_0,0),), (min(out_x_0 + output_dims[0], original_output_dims[0]),)]

            # check the validity of the partition
            if (out_bounds[1][0] - out_bounds[0][0] < 1) or (in_bounds[1][0] - in_bounds[0][0] < 1):
                raise ValueError(f"Invalid partition for spatial dimensions, please increase the split factor for layer: {layer.name}")

            if len(input_dims) > 2 :
                in_y_0 = (_ // s_x) * ((original_input_dims[1] + s_y - 1)// s_y) - overlap if s_y > 1 else 0
                in_bounds = [(max(in_x_0, 0), max(in_y_0, 0)), (min(in_x_0 + input_dims[0], original_input_dims[0]), min(in_y_0 + input_dims[1], original_input_dims[1]))]
                #check the validity of the partition
                if in_bounds[1][1] - in_bounds[0][1] < 1:
                    raise ValueError(f"Invalid partition for spatial dimensions, please increase the split factor for layer: {layer.name}")
                
            if len(output_dims) > 2:
                out_y_0 = (_ // s_x) * output_dims[1] if s_y > 1 else 0
                out_bounds = [(max(out_x_0, 0), max(out_y_0, 0)), (min(out_x_0 + output_dims[0], original_output_dims[0]), min(out_y_0 + output_dims[1], original_output_dims[1]))]
                #check the validity of the partition
                if out_bounds[1][1] - out_bounds[0][1] < 1:
                    raise ValueError(f"Invalid partition for spatial dimensions, please increase the split factor for layer: {layer.name}")


            cur = PartitionInfo(layer = layer, 
                                in_bounds = in_bounds,
                                out_bounds = out_bounds,
                                in_ch = (0, input_dims[-1]) if len(input_dims) > 2 else None,
                                out_ch = (0, output_dims[-1]) if len(output_dims) > 2 else None,
                                weights_shape = weights_dims,
                                spatial_sp_fact = partition.spatial_sp_fact if partition is not None else None, 
                                output_sp_fact = partition.output_sp_fact if partition is not None else None, 
                                input_sp_fact = partition.input_sp_fact if partition is not None else None)
            cur.set_id('spatial',  _ , partition.id if partition is not None else None)
            cur.set_split_factor('spatial', split_factor)
            new_partitions.append(cur)
        
        return new_partitions


    # If the partitions array is empty, create a new one
    if partitions is None:
        input_shape = layer.input[0].shape if type(layer.input) == list else layer.input.shape
        output_shape = layer.output.shape
        weights_dims = [w.shape for w in layer.get_weights()] #weight dimension do not change
        
        input_dims = list(input_shape[1:])
        output_dims = list(output_shape[1:])

        return _create_partitions(input_dims, output_dims, weights_dims)
        

    # If the partitions array is not empty, we apply the splitting on each of the partitions
    # new_partitions = []
    # for partition in partitions:
    #     input_bounds = partition.in_bounds
    #     output_bounds = partition.out_bounds
    #     input_dims = [input_bounds[1][i] - input_bounds[0][i] for i in range(len(input_bounds[0]))] if len(input_bounds[0]) > 1 else [input_bounds[1][0] - input_bounds[0][0]]
    #     input_dims = input_dims + [partition.in_ch] if partition.in_ch is not None else input_dims
    #     output_dims = [output_bounds[1][i] - output_bounds[0][i] for i in range(len(output_bounds[0]))] if len(output_bounds[0]) > 1 else [output_bounds[1][0] - output_bounds[0][0]]
    #     output_dims = output_dims + [partition.out_ch] if partition.out_ch is not None else output_dims
    #     weights_dims = [list(w) for w in partition.weights_shape] # weight dimensions do not change

    #     # Add the elements of the array return by _create_array to the new_partitions array
    #     new_partitions.extend(_create_partitions(input_dims, output_dims, weights_dims, partition))

    # return new_partitions


def _split_output_dims(layer, split_factor, partitions: Union[None, List[PartitionInfo]] = None):
    '''
    A function to compute the output channel partitioning of the output tensor of a layer.
    If a partitions list is given, for each object in the list, the partion strategy is applied to the object and a new list
    of partitions is created.

    Args:
    - layer : the layer to be partitioned
    - split_factor : the number of partitions to create
    - partitions : the partition list on which to apply the partitioning

    Returns:
    - a new PartitionInfo list 
    '''

    # if the does not admit channels and the split factor is greater than 1, raise an error
    if len(layer.output.shape) < 3:
        return partitions

    def _create_partitions(input_dims, output_dims, weights_dims, partition = None):
        new_partitions = []
        
        weights_dims = deepcopy(weights_dims)

        if partition is None:
            output_dims = deepcopy(output_dims)
            num_output_ch = output_dims[-1]

            in_x_0 = 0
            in_bounds = [(in_x_0,), (input_dims[0],)]
            out_x_0 = 0
            out_bounds = [(out_x_0,), (output_dims[0],)]
            if len(input_dims) > 2:
                in_y_0 = 0
                in_bounds = [(in_x_0, in_y_0), (input_dims[0], input_dims[1])]

            if len(output_dims) > 2:
                out_y_0 = 0
                out_bounds = [(out_x_0, out_y_0), (output_dims[0], output_dims[1])]
            in_ch = (0, input_dims[-1]) if len(input_dims) > 2 else None
        else:
            num_output_ch = partition.out_ch[1] - partition.out_ch[0]
            output_dims = deepcopy(partition.out_bounds) + [num_output_ch]
            in_bounds = partition.in_bounds
            out_bounds = partition.out_bounds
            in_ch = partition.in_ch

        basic_step = num_output_ch // split_factor
        additions = num_output_ch % split_factor
        partition_index = 0

        # patition the output channels and number of kernels
        while output_dims[-1] > 0:

            ch_start = partition_index * basic_step + min(partition_index, additions)
            ch_end = (partition_index + 1) * basic_step + min(partition_index + 1, additions)
            #check partition validity
            if ch_end - ch_start < 1 or ch_end > num_output_ch:
                raise ValueError("Invalid partition for output channels, please increase the split factor")
            weights_temp = []
            for i,weight in enumerate(weights_dims):
                index = -1
                weight[index] = ch_end - ch_start
                weights_temp.append(tuple(weight))

            
            

            cur = PartitionInfo(layer = layer,
                                in_bounds = in_bounds,
                                out_bounds = out_bounds,
                                in_ch = in_ch,
                                out_ch = (ch_start, ch_end),
                                weights_shape = weights_temp,
                                spatial_sp_fact = partition.spatial_sp_fact if partition is not None else None, 
                                output_sp_fact = partition.output_sp_fact if partition is not None else None, 
                                input_sp_fact = partition.input_sp_fact if partition is not None else None)
            cur.set_id('output', partition_index,  partition.id if partition is not None else None)
            cur.set_split_factor('output', split_factor)
            new_partitions.append(cur)

            output_dims[-1] -= ch_end - ch_start
            partition_index += 1


        return new_partitions


    # If the partitions array is empty, create a new one
    if partitions is None:
        input_shape = layer.input[0].shape if type(layer.input) == list else layer.input.shape # input shape does not change
        output_shape = layer.output.shape
        weights_dims = [list(w.shape) for w in layer.get_weights()]

        input_dims = list(input_shape[1:])
        output_dims = list(output_shape[1:])

        return _create_partitions(input_dims, output_dims, weights_dims)
    
    # If the partitions array is not empty, we apply the splitting on each of the partitions
    new_partitions = []
    for partition in partitions:
        input_dims = None
        output_dims = None

        weights_dims = [list(w) for w in partition.weights_shape] # weight dimensions do not change

        # Add the elements of the array return by _create_array to the new_partitions array
        new_partitions.extend(_create_partitions(input_dims, output_dims, weights_dims, partition))

    return new_partitions

    
def _split_input_dims(layer, split_factor, partitions: Union[None, List[PartitionInfo]] = None):
    """
    A function to compute the input channel partitioning of the input tensor of a layer.
    If a partitions list is given, for each object in the list, the partion strategy is applied to the object and a new list
    of partitions is created.

    Args:
    - layer : the layer to be partitioned
    - split_factor : the number of partitions to create
    - partitions : the partition list on which to apply the partitioning

    Returns:
    - a new PartitionInfo list
    """

    # if the does not admit channels and the split factor is greater than 1, raise an error
    if len(layer.input.shape) < 3:
        return partitions

    def _create_partitions(input_dims, output_dims, weights_dims, partition = None):
        new_partitions = []
        
        weights_dims = deepcopy(weights_dims)

        if partition is None:
            input_dims = deepcopy(input_dims)
            num_input_ch = input_dims[-1]
            in_x_0 = 0
            in_bounds = [(in_x_0,), (input_dims[0],)]
            out_x_0 = 0
            out_bounds = [(out_x_0,), (output_dims[0],)]
            if len(input_dims) > 2:
                in_y_0 = 0
                in_bounds = [(in_x_0, in_y_0), (input_dims[0], input_dims[1])]

            if len(output_dims) > 2:
                out_y_0 = 0
                out_bounds = [(out_x_0, out_y_0), (output_dims[0], output_dims[1])]
            out_ch = (0, output_dims[-1]) if len(output_dims) > 2 else None
        else:
            num_input_ch = partition.in_ch[1] - partition.in_ch[0]
            input_dims = deepcopy(partition.in_bounds) + [num_input_ch]
            in_bounds = partition.in_bounds
            out_bounds = partition.out_bounds
            out_ch = partition.out_ch

        basic_step = num_input_ch // split_factor
        additions = num_input_ch % split_factor
        partition_index = 0

        # partition the input channels and number of kernels
        while input_dims[-1] > 0:
            ch_start = partition_index * basic_step + min(partition_index, additions)
            ch_end = (partition_index + 1) * basic_step + min(partition_index + 1, additions)
            #check partition validity
            if ch_end - ch_start < 1 or ch_end > num_input_ch:
                raise ValueError("Invalid partition for input channels, please increase the split factor")
            weights_temp = []
            for i,weight in enumerate(weights_dims):
                index = -2 if len(weight) > 2 else 0
                weight[index] = ch_end - ch_start
                weights_temp.append(tuple(weight))
            

            cur = PartitionInfo(layer = layer,
                                in_bounds = in_bounds,
                                out_bounds = out_bounds,
                                in_ch = (ch_start, ch_end),
                                out_ch = out_ch,
                                weights_shape = weights_temp,
                                spatial_sp_fact = partition.spatial_sp_fact if partition is not None else None, 
                                output_sp_fact = partition.output_sp_fact if partition is not None else None, 
                                input_sp_fact = partition.input_sp_fact if partition is not None else None)
            cur.set_id('input', partition_index, partition.id if partition is not None else None)
            cur.set_split_factor('input', split_factor)
            new_partitions.append(cur)

            input_dims[-1] -= ch_end - ch_start
            partition_index += 1

        return new_partitions



    if partitions is None:
        input_shape = layer.input[0].shape if type(layer.input) == list else layer.input.shape # input shape does not change
        output_shape = layer.output.shape # output shape does not change
        weights_dims = [list(w.shape) for w in layer.get_weights()]

        input_dims = list(input_shape[1:])
        output_dims = list(output_shape[1:])

        return _create_partitions(input_dims, output_dims, weights_dims)
    
    # If the partitions array is not empty, we apply the splitting on each of the partitions
    new_partitions = []
    for partition in partitions:
        input_dims = None
        output_dims = None
        weights_dims = [list(w) for w in partition.weights_shape]

        # Add the elements of the array return by _create_array to the new_partitions array
        new_partitions.extend(_create_partitions(input_dims, output_dims, weights_dims, partition))

    return new_partitions

def _build_partitions_from_layer(layer, spat = 0, out_ch = 1, in_ch = 1):
    """
    A function to split the workload of a layer among different partitions
    
    Args:
    - layer : the layer to be partitioned
    - spat : 'degree' of spatial partitioning
    - out_ch : number of partitions for the output channels
    - in_ch : number of partitions for the input channels

    Returns:
    - a list of PartitionInfo objects, each representing a partition of the layer

    """

    partitions = _split_spatial_dims(layer, spat)
    partitions = _split_output_dims(layer, out_ch, partitions)
    partitions = _split_input_dims(layer, in_ch, partitions)
    for p in partitions:
        p.MACs, p.FLOPs = analyze_partition(p)
    return partitions

def _adaptive_parsel(layer):
    """
    The functions selects the best partitioning strategy for a layer, based on the layer's characteristics.
    The main objective is to create the partitions for a layer with the lowest values for the splitting factors that, at the same time,
    also allows to host the partition on a single PE. This is done to aid convergence for the mapping algorithms that will be run on the task
    graph, as well as to reduce computation at such time. 
    It is important to notice that, by splitting the network to a finer granularity, the number of partitions will increase as well as the overall 
    number of communication (even if the whole amount of data being sent over the NoC should not increase considerably). This allows the solution space
    for the mapping system to grow in dimenions, eventually allowing for more optimal solutions to be found. At the same time, the complexity of the problem
    also increases.
    The strategy chosen to split the layer into different partitons is chosen based on the layer type, as well as on its input-output-weight shapes.
    The decision on whether to stop with the partitioning is taken based on the estimated current space needed to host the values necessary for the partition
    to be computed. If the space needed is greater than the available space on a single PE, the partitioning is stopped and the partition is not created.

    Args:
    - layer : the layer to be partitioned

    Returns:
    - a tuple, representing the parameters to be used for the partitionin of the layer
    - an integer, representing the space that will be needed for the partitions of the layer
    """

    def _split_fc(layer):
        '''
        A subroutine to split the dense layers: we employ only spatial.
        
        '''
        # check the input shape of the layer
        input_shape = layer.input[0].shape if type(layer.input) == list else layer.input.shape
        output_shape = layer.output.shape

        sp_factor = 0
        # estimate the space needed to host the values (input, output, weights) of the layer on the PE
        space_needed = 0
        # input space
        space_needed += input_shape[1:].num_elements() # 8 bits per element
        # output space
        space_needed += output_shape[1:].num_elements() # 8 bits per element
        # weights space
        for w in layer.get_weights():
            space_needed += w.num_elements()

        while space_needed > PE.mem_size:
            sp_factor += 1
            space_needed = 0
            # keep the same input_shape
            input_shape = input_shape[1:].num_elements()
            # halve the output shape
            output_shape = output_shape[1:].num_elements() // 2
            # weights space
            weight_shape = input_shape * output_shape

            space_needed += input_shape + output_shape + weight_shape

        return (sp_factor, 0, 0), space_needed
    
    def _split_conv(layer):
        '''
        A subroutine to split the convolutional layers: we employ spatial and output channel partitioning
        '''
        pass

        

    # Check the type of the layer
    if isinstance(layer, (layers.InputLayer, layers.Reshape, layers.Flatten, layers.ZeroPadding1D, layers.ZeroPadding2D, layers.Identity)):
        return 0,1,1
    elif isinstance(layer, layers.Add):
        return 1,1,1
    elif isinstance(layer, (layers.ReLU, layers.ELU, layers.Activation)) and layer.activation.__name__ != 'softmax':
        return 0,1,1
    elif isinstance(layer, (layers.MaxPooling1D, layers.AveragePooling1D, layers.GlobalAveragePooling1D, layers.GlobalMaxPooling1D)):
        return 0,1,1
    elif isinstance(layer, (layers.MaxPooling2D, layers.AveragePooling2D, layers.GlobalAveragePooling2D, layers.GlobalMaxPooling2D)):
        return 0,1,1
    elif isinstance(layer, layers.BatchNormalization):
        return 0,2,1
    elif isinstance(layer, (layers.Conv1D, layers.Conv2D)):
        return 2,1,1
    elif isinstance(layer, (layers.DepthwiseConv2D, layers.DepthwiseConv1D)):
        return 1,1,1
    elif isinstance(layer, (layers.Conv1DTranspose, layers.Conv2DTranspose)):
        return 1,1,1
    elif isinstance(layer, layers.Dropout):
        return 0,1,1
    elif isinstance(layer, layers.Dense) or (isinstance(layer, layers.Activation) and layer.activation.__name__ == 'softmax'):
        return 1,1,1
    else:
        raise ValueError("Invalid layer type: {} of type {}".format(layer.name, type(layer)))
        

def _build_spatial_deps(partitions, index1, index2, deps = None):
    """
    The function builds the dependencies between the partitions of two layers based on the spatial partitioning technique:
    in particular, it takes as input the partitions list, searches for the partitions of the two - already partitioned - layers 
    and builds the dependencies among this partition.
    OSS: it is assumed that the dependency between the two layers is already established
    and correct. This must be ensured at the moment of the creation of the dependency list among the layers

    Args:
    - partitions : an object containing the partitions of the layers
    - index1: the index of the first layer in the model
    - index2: the index of the second layer in the model

    Returns:
    - a dict of dependencies between the partitions of the two layers
    """
    partitions_1 = partitions[index1.name]
    partitions_2 = partitions[index2.name]

    # we then build the dependencies between the partitions: to do so, we must look at the spatial input and ouput dimensions of the partitions
    # of the two layers and check if they 'overlap'
    deps = {} if deps is None else deps
    for p1 in partitions_1:
        for p2 in partitions_2:
            
            # check if the partitions overlap:
            if len(p1.out_bounds[0]) > 1 and len(p2.in_bounds[0]) > 1:
                # Conv layers
                if p1.out_bounds[0][0] <= p2.in_bounds[1][0] and p1.out_bounds[1][0] >= p2.in_bounds[0][0] and p1.out_bounds[0][1] <= p2.in_bounds[1][1] and p1.out_bounds[1][1] >= p2.in_bounds[0][1]:
                        overlap = (min(p1.out_bounds[1][0], p2.in_bounds[1][0]) - max(p1.out_bounds[0][0], p2.in_bounds[0][0])) * (min(p1.out_bounds[1][1], p2.in_bounds[1][1]) - max(p1.out_bounds[0][1], p2.in_bounds[0][1]))
                        if deps.get((p1.id, p2.id)) is not None:
                            deps[(p1.id, p2.id)] += overlap
                        else:
                            deps[(p1.id, p2.id)] = overlap
            else:
                # Dense layers
                # OSS: for the dense layers, depependencies are always present between the partitions of a layer and the partitions of the next layer,
                # in partitcular, the communication size is equal to the number of output neurons of the partition of the first layer
                overlap = p1.out_bounds[1][0] - p1.out_bounds[0][0]
                if deps.get((p1.id, p2.id)) is not None:
                    deps[(p1.id, p2.id)] += overlap
                else:
                    deps[(p1.id, p2.id)] = overlap
            

    return deps
     

def _build_outin_deps(partitions, index1, index2, deps = None):
    """
    A function to build the dependencies between the partitions of two layers based on the input/output channel partitioning technique.

    Args:
    - partitions : a obj containing the partitions of the layers
    - index1: the index of the first layer in the model
    - index2: the index of the second layer in the model
    - deps : a dictionary of dependencies between the partitions of the layers

    Returns:
    - a dict of dependencies between the partitions of the two layers
    """

    partitions_1 = partitions[index1.name]
    partitions_2 = partitions[index2.name]

    # we then build the dependencies between the partitions of the two layers
    deps = {} if deps is None else deps
    for p1 in partitions_1:
        for p2 in partitions_2:

            if p1.out_ch is not None and p2.in_ch is not None:
                # check for the overlap between the output channels of the first layer and the input channels of the second layer.
                # OSS: no need to check that the input channels assigned to the partition since it is assumed that, indipendently from 
                # the input channels assigned, the partial results will still need to be reduced and passed to the next layer, thus
                # creating a dependency between the partitions. Furthermore, also the number of input channels assigned to the partition
                # is not relevant in terms of communication weight, since berfore performing the computation, we can simply reduce the partial
                # results corresponding to different input channels, thus sending a number of bytes corresponding to a single channel (2D) tensor.
                if p1.out_ch[0] <= p2.in_ch[0] and p1.out_ch[1] >= p2.in_ch[0]:
                    overlap = p1.out_ch[1] - p2.in_ch[0]
                    if deps.get((p1.id, p2.id)) is not None:
                        deps[(p1.id, p2.id)] *= overlap
                    else:
                        deps[(p1.id, p2.id)] = overlap
                

    return deps


def _build_layer_deps(model: keras.Model):
    """"
    A function to infer the dependencies between the layers of a keras model: it looks at the model's dependency graph and builds
    a list of dependencies between the layers of the model. Those are then used to build the dependencies between the partitions of the layers.

    Args:
    - model : the keras model for which to infer the dependencies

    Returns:
    - a list of dependencies between the layers of the model
    """
    dependencies = []
    for layer in model.layers:
        for node in layer._inbound_nodes:
            if isinstance(node.inbound_layers, list):
                for inbound_layer in node.inbound_layers:
                    dependencies.append((inbound_layer, layer))
            else:
                inbound_layer = node.inbound_layers
                dependencies.append((inbound_layer, layer))

    return dependencies


def _build_partitions_deps(partitions, layer_to_layer_list):
    """
    Given the dictionary of partitions and the list of dependent layers, the function builds the dependencies between the partitions of the layers
    included in the layer_to_layer_list.
    
    Args:
    - partitions : a dictionary of partitions of the layers
    - layer_to_layer_list : a list of of tuples representing the dependent layers

    Returns:
    - a dictionary of dependencies between the partitions of the layers
    """

    deps = {}
    for dep in layer_to_layer_list:
        layer1 = dep[0]
        layer2 = dep[1]

        # build the dependencies between the partitions of the two layers
        deps[(layer1.name, layer2.name)] = _build_spatial_deps(partitions, layer1, layer2)
        deps[(layer1.name, layer2.name)] = _build_outin_deps(partitions, layer1, layer2, deps[(layer1.name, layer2.name)])

    return deps

def build_partitions(model, verbose = False):
    """
    The function creates the partitions for each layer in the model, based on the partitioning strategies defined above.

    Args:
    - model : the model to partition

    Returns:
    - a dict of partitions: the key is the layer name and the value is a list of PartitionInfo objects, each representing a partition of the layer
    """

    partitions = {}
    pe = PE()
    layer_deps = _build_layer_deps(model)

    # first run for the input layer
    input_layer = model.layers[0]
    spat, out_ch, in_ch = _adaptive_parsel(input_layer)
    partitions[input_layer.name] = _build_partitions_from_layer(input_layer, spat, out_ch, in_ch)
    if verbose:
        print("Layer {} partitioned succesfully with partition parameters: {} ".format(input_layer.name, (spat, out_ch, in_ch)))

    # then proceed with the other layers
    for prev_layer, layer in layer_deps:
        spat, out_ch, in_ch = _adaptive_parsel(layer)
        partitions[layer.name] = _build_partitions_from_layer(layer, spat, out_ch, in_ch)
        if verbose:
            print("Layer {} partitioned succesfully with partition parameters: {} ".format(layer.name, (spat, out_ch, in_ch)))

    # build the dependencies between the partitions
    partitions_deps = _build_partitions_deps(partitions, layer_deps)

    return partitions, partitions_deps

def plot_partitions(partitions, partitions_deps):
    """
    A function to plot the partitions of a layer using pydot package.

    Args:
    - partitions : a dictionary of partitions of the layers

    Returns:
    - a plot representing the task graph that will be deployed on the NoC
    """
    task_id = 0

    def format_node(partition: PartitionInfo):
    
        # we divide the node horizontally in 3 parts: the first part contains the partition id,
        # the second contains the input, output bounds, the number of input and output channels and the weights shape
        # the third part contains the MACs and FLOPs
        struct = f"{partition.id}\ntask_id:{partition.task_id} | layer_type:{type(partition.layer).__name__}\ninput bounds:{partition.in_bounds}\noutput bounds:{partition.out_bounds}\ninput channels:{partition.in_ch}\noutput channels:{partition.out_ch}\nweights shape:{partition.weights_shape} | MACs:{partition.MACs}\nFLOPs:{partition.FLOPs}"
        return struct

    # get the type of keras layer


    graph = pydot.Dot(graph_type='digraph')
    for layer, partition_list in partitions.items():
        for partition in partition_list:
            partition.task_id = task_id
            task_id += 1
            node = pydot.Node(partition.id,label = format_node(partition), shape = "Mrecord")
            graph.add_node(node)

    for key, value in partitions_deps.items():
        for dep, weight in value.items():
            edge = pydot.Edge(dep[0], dep[1], label = weight)
            graph.add_edge(edge)

    graph.write_png('task_graph.png')


def model_to_task_graph(model):
    """
    A function to create the depencency graph of the model that will be used for the simulation on the NoC.

    Args:
    - model : the model for which to create the dependency graph

    Returns:
    - a dependency graph of the model
    """

    dep_graph = dg.TaskGraph()
    parts, deps = build_partitions(model)

    task_id = 0
    dep_id = 10 ** math.ceil(math.log10(len(parts.items())))
    layer_id = 0
    scaling_factor = 100
    processing_time = 5

    # assign task ids to the partitions
    for layer, partitions in parts.items():
        for partition in partitions:
            if isinstance(partition.layer, layers.InputLayer):
                continue
            partition.task_id = task_id
            partition.layer_id = layer_id

            dep_graph.add_task_fully_qualified(id=partition.task_id, type = "COMP_OP", layer_id = partition.layer_id,  ct_required=partition.FLOPs//scaling_factor, dep = [])

            task_id += 1
        layer_id += 1

    for key, value in deps.items():
        partitions1 = parts[key[0]]
        partitions2 = parts[key[1]]
        for dep, weight in value.items(): 
            partition1_match = (partition for partition in partitions1 if partition.id == dep[0])
            partition2_match = (partition for partition in partitions2 if partition.id == dep[1])
            partition1 = next(partition1_match)
            partition2 = next(partition2_match)
        
            first_node ="start" if isinstance(partition1.layer, layers.InputLayer) else partition1.task_id
            print(first_node)
            second_node = partition2.task_id
            comm_type = "WRITE" if isinstance(partition1.layer, layers.InputLayer) else "WRITE_REQ"
            dep_graph.add_dependency_fully_qualified(first_node, second_node , id = dep_id, type = comm_type , size = weight//scaling_factor, pt_required= processing_time , cl = 0, dep= [-1] if isinstance(partition1.layer, layers.InputLayer) else [partition1.task_id])

            # fetch the second partition from the graph and add the dependency of the communication
            second_task = dep_graph.add_task_dep(second_node, dep_id)
            dep_id += 1

    # Finally, connect the last layer partitions to the "end" node
    for partition in parts[model.layers[-1].name]:
        dep_graph.add_dependency_fully_qualified(partition.task_id, "end", id = dep_id, type = "WRITE", size = 0, pt_required = processing_time, cl = 0, dep = [partition.task_id])
        dep_id += 1

    return dep_graph

"""
* = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = *
3. MODEL ANALYSIS FUNCTIONS: used to analyze the model and compute the FLOPs and MACs.
* = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = *
"""

import prettytable as pt

def analyze_partition(partition):
    """
    The function gets as input a PartitionInfo object and computes the number of FLOPs (and MACs if available) needed to perform the computation for a single sample

    Args:
    - partition : the partition for which we want to compute the FLOPs (and MACs)

    Returns:
    - FLOPs : the number of FLOPs performed in the partition
    """


    FLOPs = 0
    MACs = 0

    # Get the layer and the input and output shapes of the partition
    layer = partition.layer
    inputs_bounds = partition.in_bounds
    inputs_shape =[inputs_bounds[1][i] - inputs_bounds[0][i] for i in range(len(inputs_bounds[0]))] if len(inputs_bounds[0]) > 1 else [inputs_bounds[1][0] - inputs_bounds[0][0]]
    # prepend a 0 to the input shape to make it compatible to the hooks
    inputs_shape = [0] + inputs_shape
    outputs_bounds = partition.out_bounds
    outputs_shape = [outputs_bounds[1][i] - outputs_bounds[0][i] for i in range(len(outputs_bounds[0]))] if len(outputs_bounds[0]) > 1 else [outputs_bounds[1][0] - outputs_bounds[0][0]]
    # prepend a 0 to the output shape also
    outputs_shape = [0] + outputs_shape
    # Compute the FLOPs (and MACs) using the hook
    if type(layer) in register_hooks:
        FLOPs, MACs = register_hooks[type(layer)](layer, inputs_shape, outputs_shape)
        # if the partitioned layer also has an activation function, we append also those FLOPs and MACs
        if hasattr(layer, "activation"):
            activation = layer.activation.__name__
            FLOPs_act, MACs_act = register_hooks[activation](layer, inputs_shape, outputs_shape)
            MACs += MACs_act
            FLOPs += FLOPs_act

    return MACs, FLOPs

def _analyze_layer(layer,activation = None):
    '''
    The function gets as input a keras layer and computes the number of FLOPs (and MACs if available) needed to perform the computation for a single sample.

    Parameters:
    - layer : the layer for which we want to compute the FLOPs (and MACs)
    - activation : the activation function of the layer (if available)

    Returns:
    - FLOPs : the number of FLOPs performed in the layer
    - MACs : the number of MACs performed in the layer (if available)
    '''
    FLOPs = 0
    MACs = 0

    # We have to possiblityes: either the layer is a keras layer or an activation function
    if activation is not None:
        hook = register_hooks[activation]
    else:
        hook = register_hooks[type(layer)]

    if hook is not None:
        # Get the input and output shapes of the layer
        inputs_shape = layer.input_shape
        outputs_shape = layer.output_shape
        # Compute the FLOPs (and MACs) using the hook
        MACs, FLOPs = hook(layer, inputs_shape, outputs_shape)

    return MACs, FLOPs


def analyze_ops(model: keras.Model, incl_info = False):
    '''
    The function gets as input a model and computes the number of MACs and FLOPs needed to perform the computation.
    It then prints out a summary of the model, with the number of FLOPs (and MACs, when available) for each layer using the hooks defined above.

    Parameters:
    - model : the model for which we want to compute the MACs/FLOPs
    - include_info : a flag that specifies if we want to include additional information in the summary

    Returns:
    - a table with the number of MACs and FLOPs for each layer of the model
    and the total number of MACs and FLOPs for the whole model
    '''

    print("--------------------------------------------* Model " + model.name + " Parameters *--------------------------------------------")
    total_MACs = 0
    total_FLOPs = 0
    included_info = ['Input parameters (bytes)', 'Input Shape', 'Weights (bytes)', 'Weights Shape', 'Output parameters (bytes)', 'Output Shape'] if incl_info else []

    # Create a PrettyTable instance
    table = pt.PrettyTable()
    table.field_names = ["Layer", "Layer number"] + included_info + [ "FLOPs", "MACs"]
    total_parameters = 0

    # Iterate over layers and activations in the model
    for i, layer in enumerate(model.layers):
        # hook = register_hooks[type(layer)]
        FLOPs, MACs = _analyze_layer(layer)
        total_MACs += MACs
        total_FLOPs += FLOPs


        if incl_info:
            # Get the number of input parameters
            input_params = np.prod(layer.input.shape[1:]) if type(layer.input) != list else 2*np.prod(layer.input[0].shape[1:])
            input_dim = layer.input.shape[1:] if type(layer.input) != list else layer.input[0].shape[1:]
            
            # Get the number of weights
            weights = int(np.sum([np.prod(w.shape) for w in layer.get_weights()]))
            # Take into accont the bias
            if layer in [layers.Conv2D, layers.Conv1D, layers.DepthwiseConv2D, layers.DepthwiseConv1D, layers.Conv2DTranspose, layers.Conv1DTranspose] and layer.use_bias:
                weights += len(layer.get_weights())
            
            weights_dim = [w.shape for w in layer.get_weights()]
            
            # Get the number of output parameters
            output_params = np.prod(layer.output.shape[1:])
            output_dim = layer.output.shape[1:]

            
            # Add the number of weights to the total number of parameters
            total_parameters += weights + output_params + input_params

            table.add_row([layer.name, i, input_params, input_dim, weights, weights_dim, output_params, output_dim, MACs, FLOPs], divider = False if hasattr(layer, "activation") and type(layer) != layers.Activation else True)

            if hasattr(layer, "activation") and type(layer) != layers.Activation:
                activation = layer.activation.__name__
                FLOPs_act, MACs_act = _analyze_layer(layer, activation)
                total_MACs += MACs_act
                total_FLOPs += FLOPs_act
                table.add_row([activation, i,output_params, output_dim, 0, [], output_params, output_dim, MACs_act, FLOPs_act], divider = True)
            
            
        else: 
            # Add a row to the table for this layer
            table.add_row([layer.name, i, MACs, FLOPs], divider = False if hasattr(layer, "activation") and type(layer) != layers.Activation else True)

            # For each layer, also get the activation function if available and add it to the table
            if hasattr(layer, "activation") and type(layer) != layers.Activation:
                activation = layer.activation.__name__
                FLOPs_act, MACs_act = _analyze_layer(layer, activation)
                total_MACs += MACs_act
                total_FLOPs += FLOPs_act
                table.add_row([activation, i, MACs_act, FLOPs_act], divider = True)

            

    print(table)
    print(f"Total parameters: {total_parameters}")
    print(f"Total: MACs={total_MACs}, FLOPs={total_FLOPs}")
    print("------------------------------------------------------------------------------------------------------------------------")






