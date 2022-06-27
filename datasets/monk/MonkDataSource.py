import Utility
from Layer import Layer
from Hyperparameters import Hyperparameters


class MonkDataSouce:

    def __init__(self, namefile_test, namefile_train):
        columns = ["target", "a1", "a2", "a3", "a4", "a5", "a6", "id"]

        self.input_test,  self.target_test  = self.oneHotEncoding(namefile_test,  columns)
        self.original_input_train, self.original_target_train = self.oneHotEncoding(namefile_train, columns)

        self.input_test, self.target_test = Utility.shuffle_two_list_together(self.input_test, self.target_test)
        self.original_input_train, self.original_target_train = Utility.shuffle_two_list_together(self.original_input_train, self.original_target_train)


    def oneHotEncoding(self, namefile, columns):
        """ This function returns the 1-Hot Encoding using sklearn's utility """
        import pandas as pd
        from sklearn.preprocessing import OneHotEncoder
        import numpy as np

        monks = pd.read_csv(namefile, sep=" ", header=None, names=columns)
        monks.set_index('id', inplace=True)
        targets = np.array(monks.pop('target')).reshape((-1,1))
        inputs  = OneHotEncoder().fit_transform(monks).toarray().astype(np.float32)
        # Reshape 0's into -1's
        inputs_clear = np.where(inputs == 0, -1, inputs)

        return inputs_clear, targets

    def getTestSet(self):
        return self.input_test, self.target_test

    def getOriginalTrainingDataset(self):
        return self.original_input_train, self.original_target_train

    def getDatasetForValidation(self):
        return self.getOriginalTrainingDataset()

    @staticmethod
    def get_monk_1_hyperparam():
        """ Best hyperparam for MONK - 1 """
        topology = [
            Layer(17, activation='none'),
            Layer(8, activation='sigmoid'),
            Layer(1, activation='sigmoid')
        ]

        start_lr = 0.5
        L2 = 0
        alpha_momentum = 0.9
        batch_size = 1
        max_epochs = 500
        threshold_variance = 0.

        hyperparams = Hyperparameters(topology=topology, start_lr=start_lr, L2=L2,
                                      alpha_momentum=alpha_momentum, batch_size=batch_size,
                                      max_epochs=max_epochs, nesterov=False,
                                      threshold_variance=threshold_variance, lr_decay_type="no")

        return hyperparams

    @staticmethod
    def get_monk_2_hyperparam():
        """ Best hyperparam for MONK - 2 """
        topology = [
            Layer(17, activation='none'),
            Layer(4, activation='sigmoid'),
            Layer(1, activation='sigmoid')
        ]

        start_lr = 0.9
        L2 = 0
        alpha_momentum = 0.9
        batch_size = 1
        max_epochs  = 500
        threshold_variance = 0.

        hyperparams = Hyperparameters(topology=topology, start_lr=start_lr, L2=L2,
                                      alpha_momentum=alpha_momentum, batch_size=batch_size,
                                      max_epochs=max_epochs, nesterov=False,
                                      threshold_variance=threshold_variance, lr_decay_type="no")

        return hyperparams

    @staticmethod
    def get_monk_3_hyperparam():
        """ Best hyperparam for MONK - 3 """

        topology = [
            Layer(17, activation='none'),
            Layer(8, activation='sigmoid'),
            Layer(1, activation='sigmoid')
        ]

        start_lr = 0.1
        L2 = 0
        alpha_momentum = 0.5
        batch_size = 1
        max_epochs  = 250
        threshold_variance = 0.
        lr_decay_type = "linear"

        hyperparams = Hyperparameters(topology=topology, start_lr=start_lr, L2=L2,
                                      alpha_momentum=alpha_momentum, batch_size=batch_size,
                                      max_epochs=max_epochs, nesterov=False,
                                      threshold_variance=threshold_variance,
                                      lr_decay_type=lr_decay_type)

        return hyperparams

    @staticmethod
    def get_monk_3_reg_hyperparam():
        """ Best hyperparam for MONK - 3 Reg """

        topology = [
            Layer(17, activation='none'),
            Layer(8, activation='sigmoid'),
            Layer(1, activation='sigmoid')
        ]

        start_lr = 0.1
        L2 = 1.e-05
        alpha_momentum = 0.5
        batch_size = 1
        max_epochs  = 250
        threshold_variance = 0.
        lr_decay_type = "linear"

        hyperparams = Hyperparameters(topology=topology, start_lr=start_lr, L2=L2,
                                      alpha_momentum=alpha_momentum, batch_size=batch_size,
                                      max_epochs=max_epochs, nesterov=False,
                                      threshold_variance=threshold_variance, lr_decay_type=lr_decay_type)

        return hyperparams
