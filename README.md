# TeSa: TrajEctory-based and Semantic-Aware DHGNN

## Introduction

> **Motivation** Dynamic Graph Neural Networks (DGNNs) are powerful for modeling dynamic interactions, but most methods focus on **homogeneous graphs**, ignoring the complexity of **heterogeneous graphs** found in many real-world scenarios. Capturing different types of nodes and edges allows for better understanding of complex relationships and evolving semantics.
>
> **Method** We propose TeSa, a Trajectory and Semantic-aware Dynamic Heterogeneous Graph Neural Network. TeSa combines trajectory-based evolution to model individual node dynamics and semantic-aware aggregation to capture complex relationships between nodes. Using temporal point processes and multi-type edge aggregation, TeSa effectively models dynamic and heterogeneous graph structures. Our approach achieves state-of-the-art performance across multiple datasets.
>
> Paper link: [A Trajectory and Semantic-aware Dynamic Heterogeneous Graph Neural Network](https://openreview.net/pdf?id=ZD9811KEYd)

## Running the experiments

### Requirements

- python >= 3.8

- pandas==0.24.2

- torch==1.1.0

- tqdm==4.41.1

- numpy==1.16.4

- scikit_learn==0.22.1

### Command

python -u learn_edge.py -d aminer --bs 200 --uniform --n_degree 20 --agg_method attn --attn_mode prod --n_node_type 3 --n_node_type 3 --tra_len 6

### General Flags

> optional arguments:
>
>> -h, --help              show this help message and exit
>>
>> -d DATA, --data DATA    data sources to use, try yelp or aminer
>>
>> --bs BS                 batch_size
>>
>> --prefix PREFIX         prefix to name the checkpoints
>>
>> --n_degree N_DEGREE     number of neighbors to sample
>>
>> --n_head N_HEAD         number of heads used in attention layer
>>
>> --n_epoch N_EPOCH       number of epochs
>>
>> --n_layer N_LAYER       number of network layers
>>
>> --lr LR                 learning rate
>>
>> --drop_out DROP_OUT     dropout probability
>>
>> --gpu GPU               idx for the gpu to use
>>
>> --node_dim NODE_DIM     Dimentions of the node embedding
>>
>> --time_dim TIME_DIM     Dimentions of the time embedding
>>
>> --agg_method            local aggregation method
>>
>> --n_edge_type           number of edge types
>>
>> --n_node_type           number of node types
>>
>> --tra_len               number of events in the node's trajectory

## TODOs

> Refactor and structure the code to better organize the components of the TeSa model. This includes improving the implementation of trajectory-based evolution, semantic-aware aggregation, and multi-type edge handling, as well as optimizing the overall pipeline for easier experimentation and evaluation across different datasets and settings.
