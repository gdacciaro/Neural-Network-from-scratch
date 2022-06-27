import math
from random import randint

import Constants
from Hyperparameters import Hyperparameters
from Layer import Layer
from NeuralNetwork import NeuralNetwork
from datasets.cup.CupDataSource import CupDataSource
from functions.loss_functions.LossFunctionFactory import LossFunctionFactory

""" This file runs the best model 10 times, changing the seed randomly.
    In this way, we perform a random search on the seed, in order to find the one that returns the lowest loss 
    calculated on the VL set, using the MEE.
    The hyperparameter set was selected by the winner of the Fine Grid Search.
    """

def start():

    topology = [
        Layer(10, activation='none'),
        Layer(32, activation='sigmoid'),
        Layer(20, activation='sigmoid'),
        Layer(2, activation='identity'),
    ]

    start_lr = 0.11
    L2 = 0.000009
    alpha_momentum = 0.55
    max_epochs = 250
    threshold_variance = 1.e-6
    batch_size = 32
    lr_decay_type = "linear"

    dataset = CupDataSource(namefile_train="../../datasets/cup/ML-CUP21-TR.csv",
                            namefile_test="../../datasets/cup/ML-CUP21-TS.csv")

    input, target,  inputVL, targetVL = dataset.getTRAndVL_forRetraining()
    input_TS, output_TS = dataset.getInternalTestSet()

    results = []

    hypers = Hyperparameters(topology=topology, start_lr=start_lr, L2=L2,
                             alpha_momentum=alpha_momentum, batch_size=batch_size,
                             max_epochs=max_epochs, nesterov=False,
                             threshold_variance=threshold_variance, lr_decay_type=lr_decay_type)

    for index in range(0,100):
        Constants.seed = randint(0, 390810)
        print("["+str(index)+"] Random seed selected:", Constants.seed)

        nn = NeuralNetwork(hypers, log=False, forceMetrics=True, task="regression")
        try:
            nn.train(input, target, inputVL, targetVL)
        except:
            print("Exception")
            continue

        tr_loss = nn.estimate_error(input, target, LossFunctionFactory("mse"))
        vl_loss = nn.estimate_error(inputVL, targetVL, LossFunctionFactory("mse"))
        ts_loss = nn.estimate_error(input_TS, output_TS, LossFunctionFactory("mse"))
        print("[" + str(index) + "] [MSE]  Result: \n\ttr_loss=", tr_loss)
        print("\tvl_loss=", vl_loss)
        print("\tts_loss=", ts_loss)


        tr_loss = nn.estimate_error(input, target, LossFunctionFactory("mee"))
        vl_loss = nn.estimate_error(inputVL, targetVL, LossFunctionFactory("mee"))
        ts_loss = nn.estimate_error(input_TS, output_TS, LossFunctionFactory("mee"))
        print("["+str(index)+"] [MEE]  Result: \n\ttr_loss=", tr_loss)
        print("\tvl_loss=", vl_loss)
        print("\tts_loss=", ts_loss)

        print("---")

        metric = vl_loss

        object = {
            "seed": Constants.seed,
            "result": metric
        }
        results.append(object)
        index += 1

    result_min = math.inf
    seed_min = math.inf
    for item in results:
        if item["result"] < result_min:
            result_min = item["result"]
            seed_min = item["seed"]

    print("Winner: ", seed_min)

    nn = NeuralNetwork(hypers, log=True, forceMetrics=True, task="regression")
    nn.train(input, target, inputVL, targetVL)

    nn.plot()
    nn.plot_blind_targets()

if __name__ == '__main__':
    start()
