from NeuralNetwork import NeuralNetwork
from datasets.monk.MonkDataSource import MonkDataSouce

if __name__ == '__main__':

    dataset_name = "monks-1"
    # dataset_name = "monks-2"
    # dataset_name = "monks-3"
    # dataset_name = "monks-3-reg"

    if dataset_name == "monks-1":
        hyperparams = MonkDataSouce.get_monk_1_hyperparam()
    elif dataset_name == "monks-2":
        hyperparams = MonkDataSouce.get_monk_2_hyperparam()
    elif dataset_name == "monks-3":
        hyperparams = MonkDataSouce.get_monk_3_hyperparam()
    elif dataset_name == "monks-3-reg":
        hyperparams = MonkDataSouce.get_monk_3_reg_hyperparam()


    if dataset_name.startswith("monks-3"):
        dataset_name = "monks-3"

    dataset = MonkDataSouce(namefile_train="./datasets/monk/" + dataset_name + ".train",
                            namefile_test="./datasets/monk/"+dataset_name+".test")

    input, target = dataset.getOriginalTrainingDataset()
    input_TS, target_TS = dataset.getTestSet()

    nn = NeuralNetwork(hyperparams, log=True, forceMetrics=True, task="classification")
    nn.train(input, target, input_TS, target_TS)

    print("TR error: ",     nn.estimate_error(input, target))
    print("TS error:",      nn.estimate_error(input_TS, target_TS))
    print("TR accuracy:",   nn.get_accuracy(input, target))
    print("TS accuracy:",   nn.get_accuracy(input_TS, target_TS))

    nn.plot()





