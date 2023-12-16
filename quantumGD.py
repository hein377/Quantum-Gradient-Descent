import neural_network_instantiation
import neural_network_propagation
import neural_network_setup

# DATASET PREPROCESSING
# All 4 files have class variable in last column
DATA_PATH = "D:\CloudStorage\Hein\Academics\Projects\Quantum Gradient Descent\Preprocessed Datasets"

def process_data(filename):
    ls = []
    with open(filename) as f:
        for line in f:
            line = line.strip().split(",")
            inp, output = [], []
            ls.append((inp, output))
    return ls

# VARIABLES

# CLASSICAL NEURAL NETWORK

# QUANTUM NEURAL NETWORK
