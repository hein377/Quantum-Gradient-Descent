import numpy as np
from numpy import pi
import math
import random
import sys
import pandas as pd
import pickle
from fxpmath import Fxp
import matplotlib.pyplot as plt
import time
from bitstring import BitArray
from bitarray.util import int2ba
import cmath
from random import randrange
# Importing standard Qiskit libraries
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile, Aer, IBMQ
from qiskit.tools.visualization import plot_histogram, plot_state_city
from qiskit.quantum_info import Operator

# Loading your IBM Quantum account
#provider = IBMQ.enable_account("2210e8cb3b07f6c6ea71d2aefdee67a3105fec531a691d2bdcc877667abf7c24635a8250b62f9123f13b166de700ac06b696c9b84c4527951436b5aacb7bdd48")
provider = IBMQ.load_account()              #load locally saved work

'''.The testing was conducted by
setting fixed number of 5 hidden nodes, 0.7 learning 
rates, 0.3 momentum, 0.01 of target error and maximum
of 5000 epochs. The performances are measured in terms
of lowest number of epochs, lowest CPU Time and
highest accuracy
'''

#VARIABLES SET-UP
FILENAME = sys.argv[1]
TRAININGFILENAME = FILENAME[:-4] + "_training.txt"
TESTINGFILENAME = FILENAME[:-4] + "_testing.txt"
DATA_FRAME = pd.read_csv(FILENAME)
CLASS_ATTRIBUTE_NAME = DATA_FRAME.columns[-1]
NUM_COLS = len(DATA_FRAME.columns)

MAX_NUM_EPOCHS = 5000
TARGET_ACCURACY = 0.99

DIMENSION = 5+(NUM_DISTINCT_CLASS_VALS:=len(set(list(DATA_FRAME.iloc[:,-1]))))+(5*NUM_DISTINCT_CLASS_VALS)+((NUM_COLS-1)*5)
NBITS = 6                          #2 for int; 4 for frac
BIG_NBITS = 2**NBITS

# STRATIFIED RANDOM SAMPLING
def pos_values(s): return list(set(s))

def stratify(df, pos_label_vals):                   #df<pandas dataframe obj> = dataframe, pos_label_values<list>; returns [df1<dataframe>, df2<dataframe>, ...] where dfs are separated by class label
    stratified_list = []
    for pos_val in pos_label_vals: stratified_list.append(df.loc[df[df.columns[-1]] == pos_val])
    return stratified_list

def random_indices(indices, n):
    training_set_indices, testing_set_indices = indices, []
    for i in range(n): testing_set_indices.append(training_set_indices.pop(randrange(len(training_set_indices))))
    return training_set_indices, testing_set_indices

def split_training_testing(stratified_ls, testingpercent):
    training_set_ls, testing_set_ls = [], []
    for label_df in stratified_ls:
        n = int(testingpercent * len(label_df))
        training_set_indices, testing_set_indices = random_indices(list(range(len(label_df))), n)
        training_set_ls.append(label_df.iloc[training_set_indices,:])
        testing_set_ls.append(label_df.iloc[testing_set_indices,:])
    return pd.concat(training_set_ls), pd.concat(testing_set_ls)

def stratified_random_sampling(df, trainingfn, testingfn, testingpercent):
    pos_label_vals = pos_values(df[df.columns[-1]].to_numpy())
    
    stratified_list = stratify(df, pos_label_vals)
    training_set_df, testing_set_df = split_training_testing(stratified_list, testingpercent)

    training_set_df.to_csv(trainingfn)
    testing_set_df.to_csv(testingfn)
    #print_pretty_table(training_set_df, trainingfn)
    #print_pretty_table(testing_set_df, testingfn)
    
    return training_set_df, testing_set_df

training_df, testing_df = stratified_random_sampling(DATA_FRAME, TRAININGFILENAME, TESTINGFILENAME, 0.33)
training_df, testing_df = pd.read_csv(TRAININGFILENAME), pd.read_csv(TESTINGFILENAME)
training_df, testing_df = training_df.drop(training_df.columns[0], axis = 1), testing_df.drop(testing_df.columns[0], axis = 1)

def one_at_index(ind):
    ls = [0] * NUM_DISTINCT_CLASS_VALS
    ls[int(ind)-1] = 1
    return ls

def make_actual_table(df):
    actual_table = []
    for ind, row in df.iterrows():
        row = row.tolist()
        actual_table.append((np.array([row[:-1]]), np.array([one_at_index(row[-1])])))
    return actual_table

training_actual_table = make_actual_table(training_df)
testing_actual_table = make_actual_table(testing_df)

#SETUP
def print_network(network):
    for l in range(len(network)):
        print(f"layer: {l}\nweight_matrix:\n{network[l][0]}\nbias_matrix:\n{network[l][1]}\n")

def pretty_print_tt(table):
    for inputs, output in table.items():
        ls = list(inputs)
        for i in ls: print(i, end = "  ")
        print(f"|  {output}")

def create_random_wm(layer1_size, layer2_size):                 #returns matrix of shape (layer1_size, layer2_size)
    ls = []
    for row in range(layer1_size): ls.append([random.uniform(-1, 1) for i in range(layer2_size)])
    return np.array(ls)

def create_random_bv(layer_size): return np.array([[random.uniform(-1, 1) for i in range(layer_size)]])     #returns vector of shape (1, layer_size)

def initialize_network(layer_sizes):        #layer_sizes = [input_layer_size, layer1_size, layer2_size,  ..., output_layer_size ]
    network = []
    for i in range(1, len(layer_sizes)):
        size1 = layer_sizes[i-1]
        size2 = layer_sizes[i]                      #current layer size
        wm, bs = create_random_wm(size1, size2), create_random_bv(size2)
        network.append([wm, bs])
    return network

#CLASSICAL NEURAL NETWORK
def sigmoid(dot):                       #input: np.array(); returnsa np.array() w/ sigmoid function applied to each entry
    arr = []
    for i in range(np.size(dot)): arr.append(1/(1+math.exp(-dot[0][i])))
    return np.array([arr])

def find_dot(w, b, x): return (x@w) + b

def perceptron(A, dot): return A(dot)           #returns f* = A(w dot x + b)<np.array>

def calc_error(out, expected_out):                          #input: np.arrays(), same size
    return 1/(out.shape[0]) * (np.linalg.norm(expected_out - out))**2

def tuple_to_array(tup):
    ls = []
    for i in tup: ls.append(i)
    return np.array([ls])

def forward_propagate(x, network):                      #propagates through network and returns dot_vecs_ls and a_vecs_ls of; values of each layer
    a_vec = x
    dot_vecs_ls, a_vecs_ls = [np.array([[0, 0]])], [a_vec]         #don't use dot_vec[0]; just a placeholder
    for layer in network:
        w_matrix, b_scalar = layer[0], layer[1]
        dot_vec = find_dot(w_matrix, b_scalar, a_vec)
        a_vec = perceptron(sigmoid, dot_vec)
        dot_vecs_ls.append(dot_vec)
        a_vecs_ls.append(a_vec)
    return a_vec, dot_vec, dot_vecs_ls, a_vecs_ls

#BACK PROPAGATION WITH QUANTUM OPTIMIZATION
#QUANTUM GRADIENT DESCENT
def create_BinaryIntegerGridval(num_precision_bits, numbits_frac, numbits_int, signed_bit):          #returns { bit_string : (intval, grid_val) } for intval from [0, 2^n)
    return {Fxp(j, n_frac=numbits_frac, n_int=numbits_int, signed=signed_bit).bin(frac_dot=True):(j, (j/(2**num_precision_bits))-0.5+(2**(-num_precision_bits-1))) for j in range(2**num_precision_bits)}

def convertListToBinary(ls, numbits_frac, numbits_int, signed_bit): return [Fxp(element, n_frac=numbits_frac, n_int=numbits_int, signed=signed_bit).bin(frac_dot=True) for element in ls]    ##n_frac=14: 2^10=16384 so 4 decimal digits precise and n_int=10: 2^10=1024 so 3 integer digits precise; here, num_precision_bits = 14+10+1=25 bits

def create_unitary(dot):
    return sigmoid(dot) * (1-sigmoid(dot))

def create_Grid_long(num_precision_bits):       #faster for num_precision_bits>15
    ls1 = [(j/(2**num_precision_bits))-0.5+(2**(-num_precision_bits-1)) for j in range(2**(num_precision_bits-1))]
    ls2 = [-1*ls1[i] for i in range(len(ls1)-1, -1, -1)]
    return ls1+ls2

def create_Grid_short(num_precision_bits):      #faster for num_precision_bits<15
    return [(j/(2**num_precision_bits))-0.5+(2**(-num_precision_bits-1)) for j in range(2**num_precision_bits)]

def create_intgridvalPhaseunitaryval(num_precision_bits, numbits_frac, numbits_int, signed_bit):          #returns { bit_string : (intval, grid_val) } for intval from [0, 2^n)
    constant_factor = ((twoToTheN:=(2**num_precision_bits))*0.25-0.5+(twoToTheNegN:=(2**(-num_precision_bits)))*0.25)/2
    return {j:((j/twoToTheN)-0.5+(twoToTheNegN)*0.5, -j*(0.5+twoToTheNegN*0.5)+constant_factor) for j in range(2**num_precision_bits)}

INTEGER_GRIDVAL = create_intgridvalPhaseunitaryval(NBITS, 0, NBITS, False)

def initialize_quantumcircuit(num_quantumregisters, num_qubits, num_ancillary_bits, vec_dimension, network, vec_parameter, vec_parameter_binls):
    circ = [create_unitary(vec_parameter)*(num_ancillary_bits-vec_dimension)]
    for l in range(len(network)-2, -1, -1):
        delL_vector = create_unitary(vec_parameter_binls[l+1]) * (circ[0] @ (network[l+1][0]).T)
        circ = [delL_vector] + circ

    s="QuantumCircuit("
    for i in range(num_quantumregisters): s+=("QuantumRegister({}, 'qinputreg_{}'), ".format(num_qubits, i))
    s+= "ClassicalRegister({}, 'coutputreg'))".format(num_quantumregisters*num_qubits)
    return circ, eval(s)

def hadamard_transform_input_registers(circuit, num_quantumregisters, num_qubits):
    for i in range(num_quantumregisters*num_qubits): circuit.h(i)

def create_phase_oracle(cost_function, input_a, register_num, big_n, expected_out):    #register_num starts index at 0
    phase_oracle_matrix = np.zeros(shape=(big_n,big_n), dtype=np.complex_)
    for i in range(big_n): phase_oracle_matrix[i,i] = cmath.exp(2*pi*big_n*cost_function(input_a+INTEGER_GRIDVAL[i][0],expected_out)*complex(0,1))
    #for i in range(big_n): phase_oracle_matrix[i,i] = cmath.exp(2*pi*big_n*cost_function(input_a)*complex(0,1))
    return Operator(phase_oracle_matrix)

def apply_phase_oracle(circ, circuit, cost_function, input_a, register_num, n, big_n, expected_out):    #register_num starts index at 0
    phase_oracle = create_phase_oracle(cost_function, input_a, register_num, big_n, expected_out)
    circuit.append(phase_oracle, [i for i in range(register_num,register_num+n)])

#Grid Customized Inverse Quantum Fourier Transform
def qft_rotations(circuit, start_ind, end_ind, n):
    if (end_ind-start_ind) == 0: # Exit function if circuit is empty
        return circuit
    end_ind -= 1 # Indexes start from 0
    circuit.h(end_ind) # Apply the H-gate to the most significant qubit
    for qubit in range(start_ind, end_ind):
        # For each less significant qubit, we need to do a
        # smaller-angled controlled rotation: 
        circuit.cp(pi/2**(n-qubit), qubit, end_ind)

def swap_registers(circuit, start_ind, end_ind):
    for qubit in range(start_ind, end_ind//2):
        circuit.swap(qubit, end_ind-qubit-1)
    return circuit

def qft(circuit, start_ind, end_ind, n):
    # Performs QFT on start_ind to end_ind qubits in circuit
    qft_rotations(circuit, start_ind, end_ind, n)
    swap_registers(circuit, start_ind, end_ind)
    return circuit

def inverse_qft(circuit, start_ind, end_ind, n):
    # Performs inverse QFT on start_ind to end_ind qubits in circuit
    qft_circ = qft(QuantumCircuit(end_ind-start_ind), 0, end_ind-start_ind, n)      # create QFT circuit of the correct size:
    # Then we take the inverse of this circuit
    invqft_circ = qft_circ.inverse()
    # And add it to the start_ind to end_ind qubits in our existing circuit
    circuit.append(invqft_circ, circuit.qubits[start_ind:end_ind])
    return circuit.decompose() # .decompose() allows us to see the individual gates

def create_phase_unitary(big_n):
    phase_unitary_matrix = np.zeros(shape=(big_n,big_n), dtype=np.complex_)
    for i in range(big_n): phase_unitary_matrix[i,i] = cmath.exp(2*pi*INTEGER_GRIDVAL[i][1]*complex(0,1))
    return Operator(np.linalg.inv(phase_unitary_matrix))

def apply_inverse_grid_QFT(circuit, circ, register_num, d, n, big_n):
    phase_unitary = create_phase_unitary(big_n)
    circ.append(phase_unitary, [i for i in range(register_num,register_num+n)])
    inverse_qft(circ, 0, d*n, n)
    circ.append(phase_unitary, [i for i in range(register_num,register_num+n)])

def choose_best_count(counts): return sorted(counts.items(), key=lambda x:x[1], reverse=True)[0][0]

def calc_quantum_gradient(d, n, big_n, a_vec, dot_vec, dot_vecs_ls, a_vecs_ls, expected_output, network):
    circ, circuit = initialize_quantumcircuit(d, n, expected_output, a_vec, network, dot_vec, dot_vecs_ls)
    hadamard_transform_input_registers(circuit, d, n)
    circuit.barrier()
    for register_num in range(d): apply_phase_oracle(circ, circuit, calc_error, a_vec, register_num, n, big_n, expected_output)
    circuit.barrier()
    for register_num in range(d): apply_inverse_grid_QFT(circ, circuit, register_num, d, n, big_n)
    circuit.barrier()
    # Transpile for simulator
    simulator = Aer.get_backend('qasm_simulator')
    circuit = transpile(circuit, simulator)
    for i in range(d*n): eval(f"circuit.measure({i},{i})")
    # Run and get counts
    num_shots = 8192
    result = simulator.run(circuit, shots=num_shots).result()
    countstates = choose_best_count(result.get_counts(circuit))
    return circ, countstates

def choose_best(ls):                    #returns index with highest value in ls
    return ls.index(max(ls))

def test_network(network, actual_table):
    num_correct, num_instances = 0, len(actual_table)
    for tup in actual_table:
        x, expected_out = tup
        if(choose_best(forward_propagate(x, network)[0][0].tolist()) == choose_best(expected_out[0].tolist())): num_correct += 1
    return num_correct/num_instances

def back_propagation(actual_table, network):            #network = [ layer1, layer2, ... ]; layer1 = [ weight_matrix, bias_scalar ]              #actual_table = [ ( input_array <1x784 np.array of ints>, output_array <1x10 np.array of ints> ), ... ]
    lamb, num_epoch, total_testing_time = 0.7, 0, 0
    while(num_epoch < MAX_NUM_EPOCHS):
        for tup in actual_table:                        #actual_talbe = [ ( (instance_attributevals) , class), ... ]
            x, expected_output = tup
            a_vec, dot_vec, dot_vecs_ls, a_vecs_ls = forward_propagate(x, network)                            #a_vec = final layer's output perceptron, dot_vec = final layer's dot

            delL_ls, temp_layer = calc_quantum_gradient(DIMENSION, NBITS, BIG_NBITS, a_vec, dot_vec, dot_vecs_ls, a_vecs_ls, expected_output, network)

            for l in range(len(network)):
                layer = network[l]
                layer[1] = layer[1] + np.array([[lamb]]) * delL_ls[l]                               #update bias
                layer[0] = layer[0] + np.array([[lamb]]) * ((a_vecs_ls[l]).T @ delL_ls[l])              #update weight

        test_start_time = time.perf_counter()
        cur_network_accuracy = test_network(network, actual_table)
        test_end_time = time.perf_counter()
        total_testing_time += test_end_time-test_start_time
        if(cur_network_accuracy > TARGET_ACCURACY): return network, total_testing_time
        lamb *= 0.99
        num_epoch += 1
    return network, total_testing_time, num_epoch

neural_network = initialize_network([NUM_COLS-1, 5, NUM_DISTINCT_CLASS_VALS])         #layer_sizes = [#attributes_without_class, 5, #distinct_class_values]
total_train_start = time.perf_counter()
trained_network, total_testing_time, num_epochs = back_propagation(training_actual_table, neural_network)
total_train_end = time.perf_counter()
actual_training_time = ((total_train_end-total_train_start) - total_testing_time)
print(f"Time took to converge: {actual_training_time}")
print(f"Number of epochs took to converge: {num_epochs}")
print(f"Testing accuracy: {test_network(testing_actual_table, trained_network)}")