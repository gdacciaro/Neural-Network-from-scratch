import Constants
from Layer import Layer
from NeuralNetwork import NeuralNetwork
from datasets.cup.CupDataSource import CupDataSource
from functions.loss_functions.LossFunctionFactory import LossFunctionFactory
from Hyperparameters import Hyperparameters

if __name__ == '__main__':
    topology = [
        Layer(10, activation='none'),
        Layer(32, activation='sigmoid'),
        Layer(20, activation='sigmoid'),
        Layer(2, activation='identity'),
    ]

    start_lr = 0.11
    L2 = 0.000009
    alpha_momentum = 0
    max_epochs = 400
    threshold_variance = 1.e-6
    batch_size = 32

    Constants.seed = 130401 # We found this seed, doing the retraining

    dataset = CupDataSource(namefile_train="./datasets/cup/ML-CUP21-TR.csv",
                            namefile_test="./datasets/cup/ML-CUP21-TS.csv")

    input, target, input_VL, target_VL = dataset.getTRAndVL_forRetraining()
    input_TS, target_TS = dataset.getInternalTestSet()
    input_BTS = dataset.getBlindTestSet()

    hypers = Hyperparameters(topology=topology, start_lr=start_lr, L2=L2,
                             alpha_momentum=alpha_momentum, batch_size=batch_size,
                             max_epochs=max_epochs, nesterov=False, threshold_variance=threshold_variance,
                             lr_decay_type="no")
    nn = NeuralNetwork(hypers, log=True, forceMetrics=True, task="regression")
    nn.train(input, target, input_VL, target_VL)

    mse = LossFunctionFactory("mse")
    mee = LossFunctionFactory("mee")

    predictions = []
    well_print_input = []
    output_list = []

    theList = list()
    for index_input_bts, single_input_bts in enumerate(input_BTS):
        output = nn.feedforward(single_input_bts)
        theList.append((output[0], output[1]))
        nn.plotter.addBlindTarget(output[0], output[1])

    CupDataSource.write_results_on_csv(theList)

    nn.plot()
    nn.plot_blind_targets()

    tr_loss = nn.estimate_error(input, target, mse)
    vl_loss = nn.estimate_error(input_VL, target_VL, mse)
    ts_loss = nn.estimate_error(input_TS, target_TS, mse)
    print("[MSE]  Result: \n\tTR Loss=", tr_loss)
    print("\tVL Loss=", vl_loss)
    print("\tTS Loss=", ts_loss)

    tr_loss = nn.estimate_error(input, target, mee)
    vl_loss = nn.estimate_error(input_VL, target_VL, mee)
    ts_loss = nn.estimate_error(input_TS, target_TS, mee)
    print("[MEE]  Result: \n\tTR Loss=", tr_loss)
    print("\tVL Loss=", vl_loss)
    print("\tTS Loss=", ts_loss)