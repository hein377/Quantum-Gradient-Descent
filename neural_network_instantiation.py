import numpy as np
import random

# NEURAL NETWORKS CREATION
def print_network(network):
    for l in range(len(network)):
        print(f"layer: {l}\nweight_matrix:\n{network[l][0]}\nbias_matrix:\n{network[l][1]}\n")

def create_random_wm(layer1_size, layer2_size):            #returns layer2_size x layer1_size matrix
    ls = []
    for row in range(layer2_size): ls.append([random.uniform(-1, 1) for i in range(layer1_size)])
    return np.array(ls)

def create_random_bs(layer_size): return np.array([[random.uniform(-1, 1) for i in range(layer_size)]]).T       #returns layer_size x 1

def create_network(layer_sizes):        #layer_sizes = [input_layer_size, layer1_size, layer2_size, ... , output_layer_size]
    network = []
    for i in range(1, len(layer_sizes)):
        size1 = layer_sizes[i-1] 
        size2 = layer_sizes[i]                      #current layer size
        wm, bs = create_random_wm(size1, size2), create_random_bs(size2)
        network.append([wm, bs])
    return network