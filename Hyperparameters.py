import numpy as np

from Layer import Layer


class Hyperparameters:

    def __init__(self, topology, start_lr, L2, alpha_momentum, nesterov, batch_size, lr_decay_type, max_epochs, threshold_variance):

        self.topology = topology
        self.start_lr = start_lr
        self.L2 = L2
        self.alpha_momentum = alpha_momentum
        self.batch_size = batch_size
        self.nesterov = nesterov
        self.lr_decay_type = lr_decay_type
        self.max_epochs = max_epochs
        self.threshold_variance = threshold_variance

    def __str__(self):
        return "[topology=" + str(self.topology) \
               + ",\n\t\t\t\t start_lr=" + str(self.start_lr) + \
               ",\n\t\t\t\t L2=" + str(self.L2) + \
               ",\n\t\t\t\t nesterov=" + str(self.nesterov) + \
               ",\n\t\t\t\t alpha_momentum=" + str(self.alpha_momentum) + \
               ",\n\t\t\t\t batch_size=" + str(self.batch_size) + \
               ",\n\t\t\t\t lr_decay_type=" + str(self.lr_decay_type) + \
               ",\n\t\t\t\t max_epochs=" + str(self.max_epochs) + "] "

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def get_fine_param(param):
        """
        :param param: a specific value for an hyperparameter
        :return: a small range of this values
        """
        positive_fine_ratio = 1.1
        negative_fine_ratio = 0.9
        return [np.round(param * negative_fine_ratio, 9),
                param,
                np.round(param * positive_fine_ratio, 9)]

    @staticmethod
    def get_fine_param_batch_size(param):
        """
        :param param: a specific value for the hyperparameter "batch size"
        :return: a small range of this value
        """
        return [max(1, param - 10),
                param,
                int(param + 10)]

    @staticmethod
    def get_default_hyperparameters(datasetname):
        """
        :param datasetname: dataset
        :return: set of hyperparametric values for Model Selection
        """
        topology_1 = [
            Layer(17, activation='none'),
            Layer(12, activation='sigmoid'),
            Layer(1, activation='sigmoid')
        ]

        topology_2 = [
            Layer(17, activation='none'),
            Layer(8, activation='sigmoid'),
            Layer(1, activation='sigmoid')
        ]

        topology_3 = [
            Layer(17, activation='none'),
            Layer(4, activation='sigmoid'),
            Layer(1, activation='sigmoid')
        ]

        if datasetname == "monks-1" or datasetname == "monks-2" or datasetname == "monks-3":
            return {
                "lr": [0.9, 0.5, 0.1, 0.01, 0.001],
                "alpha_momentum": [0.9, 0.75, 0.5],
                "L2": [0],
                "batch_size": [1, 62, "full"],
                "lr_decay_type": ["linear", "exponential"],
                "topologies": [topology_1, topology_2, topology_3],
            }
        elif datasetname == "monks-3-reg":
            return {
                "lr": [0.1],
                "alpha_momentum": [0.5],
                "L2": [ 1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5 ],
                "batch_size": [1],
                "lr_decay_type": ["linear"],
                "topologies": [topology_2],
            }
        else:
            import Utility
            topologies = Utility.generate_topologies_for_cup_model_selection()

            return {
                "lr": [ 0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001],
                "alpha_momentum": [ 0.5, 0.75, 0.9 ],
                "L2": [0.1,0.01, 0.001, 0.0001, 0.00001],
                "batch_size": [1, 16, 32, 64,  "full"],
                "lr_decay_type": ["linear"],
                "topologies": topologies
            }

        pass