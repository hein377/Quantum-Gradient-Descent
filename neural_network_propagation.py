import numpy as np
import neural_network_setup
import pickle

def print_epoch_info(epoch_count, network_accuracy):
    print(f"EPOCH {epoch_count - 1}")
    print(f"Accuracy: {network_accuracy}")
    print("\n---------------------------\n")

'''
actual_table: table with classes as its last column; used to train network <[#instances, #attributes+1] numpy array>
network: the neural network <numpy array>
lamb: starting lambda val <double>
num_epochs: maximum # of epochs used to terminate while loop <int>
error_threshold: desired maximum error to terminate while loop <double>
activation_function: desired activation function for network <function>
activation_function_prime: derivative of desired activation function <function>
test_network: method to gauge accuracy of network on actual_table <function>
verbose: will print information if true <boolean>
'''

def back_propagation_multilayer(actual_table, network, num_epochs, lamb, error_threshold, activation_function, activation_function_prime, test_network, verbose):            #network = [ layer1, layer2, ... ]; layer1 = [ weight_matrix, bias_scalar ]              #actual_table = [ ( input_array <1x784 np.array of ints>, output_array <1x10 np.array of ints> ), ... ]
    network_error, _ = test_network(network, actual_table, activation_function)
    if verbose: print(f"Initial network error: {network_error}")
    epoch_count = 0
    while(epoch_count < num_epochs and network_error > error_threshold):
        for tup in actual_table:
            x, expected_output = tup
            #Forward Propagate
            a_vec, dot_vec, dot_vecs_ls, a_vecs_ls = neural_network_setup.forward_propagate(x, network, activation_function)                            #a_vec = final layer's output perceptron, dot_vec = final layer's dot
            #Backward Propagate - Calculate Gradient Descent Values
            delL_ls = [activation_function_prime(dot_vec)*(expected_output-a_vec)]                          #del_N           #del_Ls[-1] = delN (gradient function for LAST FUNCTION)
            for l in range(len(network)-2, -1, -1):
                delL_vector = activation_function_prime(dot_vecs_ls[l+1]) * ((network[l+1][0]).T @ delL_ls[0])
                delL_ls = [delL_vector] + delL_ls
            #Backward Propagate - Update Values
            for l in range(len(network)):
                layer = network[l]
                layer[1] = layer[1] + np.array([[lamb]]) * delL_ls[l]                               #update bias
                layer[0] = layer[0] + np.array([[lamb]]) * (delL_ls[l] @ (a_vecs_ls[l]).T)              #update weight
        epoch_count += 1
        network_accuracy, _ = test_network(network, actual_table, activation_function)              #second returned value is misclassified_ls
        lamb *= 0.999
        if verbose: print_epoch_info(epoch_count, network_accuracy)
    return network
