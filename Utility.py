import random
import numpy as np
import Constants

def GlorotInitialization(topology):
    """ This utility method initializes the list of matrices containing all the weights of the network """
    np.random.seed(Constants.seed)

    # weight_matrices_list is a list of matrices, one matrix for each layer, starting from the input layer
    weight_matrices_list = []
    # dimension of each matrix= (number of units of the next layer)*(number of units of the current layer +1)
    # +1 due to the presence of the biases (the last column contains the biases)
    for i in range(len(topology) - 1):
        fan_in= topology[i]
        fan_out= topology[i+1]
        variance = 2.0 / (fan_in + fan_out)
        stddev = np.sqrt(variance)
        # We initialize each weight with a small Gaussian value with mean = 0.0
        # and variance based on the fan-in and fan-out of the weight
        matrix= np.random.normal(loc=0.0, scale=stddev, size=(fan_out, fan_in + 1))
        #0 for biases
        matrix[:, fan_in] = 0
        weight_matrices_list.append(matrix)

    return weight_matrices_list

def UniformRandomInitialization(topology):
    """ This utility method generates the list of matrices containing all the weights of the network """
    np.random.seed(Constants.seed)

    # weight_matrices_list is a list of matrices, one matrix for each layer, starting from the input layer
    weight_matrices_list = []
    # dimension of each matrix= (number of units of the next layer)*(number of units of the current layer +1)
    # +1 due to the presence of the biases (the last column contains the biases)
    # we fill each matrix with small random values
    for i in range(len(topology) - 1):
        value = topology[i]
        nextValue = topology[i + 1]
        weight_matrices_list.append(np.random.uniform(low=Constants.dataset_interval_start,
                                                      high=Constants.dataset_interval_end,
                                                      size=(nextValue, value + 1)))
    return weight_matrices_list

def shuffle_two_list_together(list_1, list_2):
    """ Given two lists with the same length, this function mixes these
        lists while keeping the initial match between each pair """
    tmp = list(zip(list_1, list_2))
    random.seed(Constants.seed)
    random.shuffle(tmp)
    list_1, list_2= zip(*tmp)
    return list_1, list_2

def is_last_layer(layer, topology):
    return layer == len(topology) - 1

def sum_matrices_lists(matrix_A, matrix_B):
    if len(matrix_A) == 0:
        return matrix_B
    if len(matrix_B) == 0:
        return matrix_A
    else:
        for index in range(len(matrix_A)):
            matrix_A[index] = np.add(matrix_A[index], matrix_B[index])
    return matrix_A

def well_print_matrix(matrix):
    """ Given a matrix, this function prints it on the console nicely """
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))

def apply_classification_threshold(y):
    if y > Constants.classification_threshold:
        result = 1
    else:
        result = 0
    return result

def generate_topologies_for_cup_model_selection():
    """ This function provides the set of topologies for the Grid Search """
    hidden_layers = [ 2 , 3 , 4 ]
    hidden_units =  [ 16, 32, 64 ]
    topologies = []

    from Layer import Layer

    for layer in hidden_layers:
        for units in hidden_units:
            topology = [Layer(10, activation='none')]
            for number_of_hidden_layer in range(layer):
                topology.append(Layer(units, activation='sigmoid'))
            topology.append(Layer(2, activation='identity'))
            topologies.append(topology)

    return topologies