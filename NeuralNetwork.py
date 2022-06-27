import math

import numpy as np

import Metrics
import Utility
from functions.activaction_functions.ActivationFunctionUtilities import generateActivationFunctionsList
from functions.loss_functions.LossFunctionFactory import LossFunctionFactory
from functions.lr_decay_functions.LearningRateDecayFactory import LearningRateDecayFactory
from graphs.Plotter import Plotter


class NeuralNetwork(object):

    def __init__(self, hyperparameters, log=False, forceMetrics=False, task="classification", loss="mse"):
        self.isClassification = task == "classification"
        # net_list is a list of array, one array for each layer, starting from the input layer
        # each element of the array is the net input for a node of the layer
        self.net_list = []
        # out_list is a list of array, one array for each layer, starting from the input layer
        # each element of the array is the output unit for a node of the layer
        self.out_list = []

        # Hyperparams
        self.hyperpars = hyperparameters
        self.lr = hyperparameters.start_lr
        self.L2 = hyperparameters.L2
        self.nesterov = hyperparameters.nesterov
        self.alpha_momentum = hyperparameters.alpha_momentum
        self.batch_size = hyperparameters.batch_size
        self.topology = hyperparameters.topology
        self.threshold_variance = hyperparameters.threshold_variance
        self.max_epochs = hyperparameters.max_epochs
        self.original_lr = hyperparameters.start_lr

        self.units_for_layer = self.__getLayers()

        for number_of_nodes in self.units_for_layer:
            # We add +1 to the length of each layer, due to the presence of the bias
            self.out_list.append(np.ones(number_of_nodes + 1))
            self.net_list.append(np.ones(number_of_nodes + 1))

        # weight_matrices_list is a list of matrices, one matrix for each layer, starting from the input layer
        # the element [i,j] of each matrix is the weight from unit j to unit i
        self.weight_matrices_list = Utility.GlorotInitialization(self.units_for_layer)

        self.lastDeltas = [] # Needed for momentum
        for i in self.weight_matrices_list:
            self.lastDeltas.append(np.zeros(shape=i.shape))

        self.last_big_delta_nesterov = None  # Needed for Nesterov

        # list of activation functions, one for each layer, starting from the first layer (which is always 'none')
        self.activation_functions = generateActivationFunctionsList(self.__getActivationFunctions())

        # we can choose the loss function from a list of possible ones
        self.loss_function = LossFunctionFactory(loss)
        self.lrDecayObject = LearningRateDecayFactory(hyperparameters.lr_decay_type)
        self.log = log
        self.forceMetrics = forceMetrics
        self.max_non_decrescent_epochs = 15 # Early Stopping
        self.plotter = Plotter()

        if self.log:
            print("========================================================")
            print("Init of Neural Network")
            print("Hyperparams:", hyperparameters)
            print("========================================================")

    def feedforward(self, nn_input):
        """ we initialize the first array of out_list with the input units of the network """
        self.out_list[0][:-1] = np.array(nn_input)

        index_layer = 0
        for weight_matrix in self.weight_matrices_list:  # for each weight matrix
            f_act = self.activation_functions[index_layer + 1].getFunction  # code readability

            for j, nextLayersNode_enteringWeights in enumerate(weight_matrix):
                self.net_list[index_layer + 1][j] = np.dot(self.out_list[index_layer], nextLayersNode_enteringWeights)
                self.out_list[index_layer + 1][j] = f_act(self.net_list[index_layer + 1][j])
            index_layer += 1

        """ we return the output units of the output layer """
        return self.out_list[-1][:-1]

    def __initialize_small_deltas(self):
        # small_deltas is a list of arrays, one array for each layer, starting from the input layer
        # each element of the array is the error signal for a node of the layer
        small_deltas = []
        for layer in range(0, len(self.units_for_layer)):
            # we initialize the arrays to zero
            small_deltas.append(np.zeros(self.units_for_layer[layer]))

        return small_deltas

    def __initialize_big_deltas(self):
        # big_deltas is a list of matrices, one matrix for each layer, starting from the input layer
        # the element [t, u] of each matrix is the Delta for the weight from unit u to unit t
        big_deltas = []
        # dimension of each matrix= (number of units of the next layer)*(number of units of the current layer +1)
        # +1 due to the presence of the biases
        # we initialize the matrices to zero
        for m in range(1, len(self.units_for_layer)):
            big_deltas.append(np.zeros(shape=(self.units_for_layer[m], self.units_for_layer[m - 1] + 1)))
        return big_deltas

    def __get_small_delta_output_layer(self, target, output, net, activationFunction):
        """ This function calculates the error signal for a single node of the output layer,
            formula: (d_k − d_k) · f'(net_k)"""

        # partial derivative of the loss respect to the output of the output layer
        loss_function_prime = self.loss_function.getFirstDerivative(target, output)
        # first derivative of the activation function applied to the net
        act_function_prime = activationFunction.getFirstDerivativeFunction(net)

        return np.dot(loss_function_prime, act_function_prime)

    @staticmethod
    def __get_small_delta_hidden_layer(succeeding_layer_list_of_small_deltas, exiting_weights_array, net, activationFunction):
        """ This function calculates the error signal for a single node of the hidden layers,
             formula: sum from k=1 to K [ (d_k · w_k_j) ] · f_j'(net_j) """

        sum_of_exiting_deltas = 0

        # we sum the error signals of the succeeding layer multiplied by the exiting weights from the current node
        for k, exiting_small_delta in enumerate(succeeding_layer_list_of_small_deltas):
            sum_of_exiting_deltas += exiting_small_delta * exiting_weights_array[k]
        act_function_prime = activationFunction.getFirstDerivativeFunction(net)
        return np.dot(sum_of_exiting_deltas, act_function_prime)

    def __backpropagation(self, target):
        """ This function is the first step of Back-propagation Algorithm """
        target_np_array = np.array(target, dtype=float)  # array of targets
        small_deltas = self.__initialize_small_deltas()  # error signals
        big_deltas = self.__initialize_big_deltas()  # Deltas of the Generalized Delta Rule

        # we have to proceed in reverse, from the output layer to the first hidden layer
        for index_layer, layer in enumerate(reversed(range(1, len(self.units_for_layer)))):
            # we take the activation function of the current layer

            activationFunction = list(reversed(self.activation_functions))[index_layer]
            # for every node in the layer
            for node in range(self.units_for_layer[layer]):

                # case of the output layer
                if Utility.is_last_layer(layer, self.units_for_layer):
                    # error signal of a node in the output layer
                    small_delta = self.__get_small_delta_output_layer(target=target_np_array[node],
                                                                    output=self.out_list[layer][node],
                                                                    net=self.net_list[layer][node],
                                                                    activationFunction=activationFunction)
                else:  # case of hidden layers
                    # we extract the column of exiting weights from the current node
                    exiting_weights_from_node = self.weight_matrices_list[layer][:, node]
                    # error signal of a node in a hidden layer
                    small_delta = self.__get_small_delta_hidden_layer(
                                                        succeeding_layer_list_of_small_deltas=small_deltas[layer + 1],
                                                        exiting_weights_array=exiting_weights_from_node,
                                                        net=self.net_list[layer][node],
                                                        activationFunction=activationFunction)

                # to find the gradient that I will need to use to modify the array of weights,
                    # I multiply the small delta by the out of the previous layer
                big_delta = np.dot(small_delta, self.out_list[layer - 1])

                # Saving the small delta and the newly found big delta
                big_deltas[layer - 1][node] = big_delta
                small_deltas[layer][node] = small_delta

        return big_deltas

    def __update(self, big_deltas):
        """ This function is the Update rule of Back-propagation Algorithm """

        batch_lr = self.lr / self.batch_size

        for index_matrix_deltas, delta_matrix in enumerate(big_deltas):
            weights = self.weight_matrices_list[index_matrix_deltas]
            # L2
            L2_reg = np.dot(-self.L2, weights)

            # Learning Rule
            delta_w_new = np.dot(-batch_lr, delta_matrix)
            weights = np.add(weights, delta_w_new)

            # Momentum
            delta_w_old = self.lastDeltas[index_matrix_deltas]
            momentum = np.dot(delta_w_old, self.alpha_momentum)
            weights = np.add(weights, momentum)
            self.lastDeltas[index_matrix_deltas] = delta_w_new + momentum

            # L2
            weights = np.add(weights, L2_reg)

            # Update
            self.weight_matrices_list[index_matrix_deltas] = weights

    def __manage_lr_decay(self, epoch):
        self.lr = self.lrDecayObject.getNextLearningRate(actual_lr=self.lr,
                                                         initial_lr=self.original_lr,
                                                         actual_epoch=epoch)

    def __apply_nesterov(self, ultimo_big_delta):
        # (1) Apply momentum:
        #       w_new = w_old + alpha *  delta_w_old
        for index_matrix_deltas, delta_matrix in enumerate(ultimo_big_delta):
            weights = self.weight_matrices_list[index_matrix_deltas]
            delta_w_old = self.lastDeltas[index_matrix_deltas]
            momentum = np.dot(delta_w_old, self.alpha_momentum)
            weights = np.add(weights, momentum)
            self.weight_matrices_list[index_matrix_deltas] = weights

    def __train_single_epoch(self, input, target, batch_size, epoch):
        """ This function execute one single epoch,
            here are implemented the logic for batch, the mini-batch and the stocastich scenario """

        loss_list = []
        batch_index = 0

        while batch_index < len(input):
            batch_input  = input[batch_index:    batch_index + batch_size]
            batch_target = target[batch_index:   batch_index + batch_size]
            batch_input, batch_target = Utility.shuffle_two_list_together(batch_input, batch_target)

            big_delta = []
            if self.nesterov and self.last_big_delta_nesterov is not None:
                self.__apply_nesterov(self.last_big_delta_nesterov)

            for single_input, single_target in zip(batch_input, batch_target):
                """ For each input within the batch, calculate the output, the loss 
                        and its result to big_delta """
                output = self.feedforward(single_input)

                loss = self.loss_function.getFunction(single_target, output)
                loss_list.append(loss)

                big_delta_part = self.__backpropagation(single_target)
                big_delta = Utility.sum_matrices_lists(big_delta, big_delta_part)

            """ Once we have calculated the whole big_delta, we can update the weigth's matrix """
            self.__update(big_delta)

            self.last_big_delta_nesterov = big_delta
            batch_index += batch_size

        """ Once every epoch, we decay the learning rate, if needed """
        self.__manage_lr_decay(epoch)

        """ The result of the epoch is the mean of the losses of each batch"""
        return np.mean(loss_list)

    def train(self, input, target, input_to_test=None, target_to_test=None):
        import time
        mee = None
        epoch = 0
        early_stopping_counter = 0
        last_epoch_error = math.inf
        start_tot = time.time()
        is_early_stopping_active = self.max_non_decrescent_epochs > 0
        is_test_datasets_empty = input_to_test is None and target_to_test is None

        if self.batch_size == "full":
            self.batch_size = len(input)
        else:
            self.batch_size = int(self.batch_size)

        errors = []
        while True:
            start_epoch = time.time()

            loss = self.__train_single_epoch(input, target, self.batch_size, epoch)
            self.plotter.addTrainingLoss(loss)

            """ if the loss diverges, we can (sadly) stop the computation """
            if math.isnan(loss):
                break

            """ Since, compute metrics could be expensive, we calculate them once every 10 epochs """
            if epoch % 10 == 9 or self.forceMetrics:
                """ Metrics """
                if self.isClassification:
                    if input_to_test is not None and target_to_test is not None:
                        accuracy_TS = Metrics.getMetrics(self, input_to_test, target_to_test)["accuracy"]
                        self.plotter.addTestAccuracy(accuracy_TS)
                        accuracy_TR = Metrics.getMetrics(self, input, target)["accuracy"]
                        self.plotter.addTrainingAccuracy(accuracy_TR)
                else:
                    if input_to_test is not None and target_to_test is not None:
                        mee_ts = Metrics.getMetricsCup(self, input_to_test, target_to_test)
                        self.plotter.addMeeTest(mee_ts)
                        mee_tr = Metrics.getMetricsCup(self, input, target)
                        self.plotter.addMeeTrain(mee_tr)

            if self.log:
                if self.isClassification:
                    metrics = "Accuracy: [TR] "+str(accuracy_TR)+" [TS]  "+str(accuracy_TS)
                else:
                    metrics = "MEE:  [TR] "+str(mee_ts)+" [TS]  "+str(mee_tr)

                print("Epoch: ", epoch, "/", self.max_epochs, "\t| Loss: ", loss, "\t | ", np.round(time.time() - start_epoch, 6), "sec",  "\t | LR:", self.lr,"\t | ", metrics, "\t")

            """ Early stopping criterias """
            if not is_test_datasets_empty and is_early_stopping_active:
                """ Calculate the loss on the Validation Set """
                loss_on_test = self.estimate_error(input_to_test, target_to_test)
                self.plotter.addTestingLoss(loss_on_test)
                errors.append(loss_on_test)

                """ We count how many epochs have a non-decreasing loss """
                if last_epoch_error < loss_on_test:
                    early_stopping_counter += 1
                else:
                    early_stopping_counter = 0
                    last_epoch_error = loss_on_test

                max_non_decrescent_epochs_criteria = (early_stopping_counter >= self.max_non_decrescent_epochs)

                """ if the loss is always decreasing, but decreases by an infinitesimal factor,
                    we still stop earlier due to variance """
                variance = np.var(errors[-20:])

                early_stopping_criteria = 0 < variance < self.threshold_variance or \
                                          max_non_decrescent_epochs_criteria or \
                                          (epoch >= self.max_epochs)
            else:
                early_stopping_criteria = epoch >= self.max_epochs

            epoch += 1

            if early_stopping_criteria:
                break

        if self.log and not is_test_datasets_empty:
            last_epoch_error = self.estimate_error(input_to_test, target_to_test)
            printed_time = "Time: " + str(np.round(time.time() - start_tot, 6)) + "sec"
            print("Last error: ", last_epoch_error, " | ", printed_time)


    def plot(self):
        if self.isClassification:
            self.plotter.plot(["training_accuracy", "training_loss", "test_accuracy", "test_loss"])
        else:
            self.plotter.plot(["test_mee","training_loss","train_mee", "test_loss"])

    def __getLayers(self):
        result = []
        for layer in self.topology:
            result.append(int(layer.nodes))
        return result

    def __getActivationFunctions(self):
        """ This function returns the list of network activation functions """
        result = []
        for layer in self.topology:
            result.append(str(layer.activation))
        return result

    def estimate_error(self, inputs, targets, loss_function = None):
        """ This function estimates the error, given inputs and target.
            The loss function can be modified.
            """
        if loss_function is None:
            loss_function = self.loss_function
        loss = 0

        for input, target in zip(inputs, targets):
            output = self.feedforward(input)
            single_loss = loss_function.getFunction(predicted=output, target=target)
            loss += single_loss

        loss /= len(inputs)

        return loss

    def plot_blind_targets(self):
        self.plotter.plot_blind_target()

    def get_accuracy(self, inputs, targets):
        accuracy = self.get_metrics(inputs, targets)["accuracy"]
        return accuracy

    def get_metrics(self, inputs, targets):
        return Metrics.getMetrics(self, inputs, targets, log=False)