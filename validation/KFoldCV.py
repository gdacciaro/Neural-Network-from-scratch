import math
import multiprocessing as mp

import numpy as np

from validation.distribuited_computing import kfold_distributed_computing_cup
from NeuralNetwork import NeuralNetwork
from datasets.cup.CupDataSource import CupDataSource
from datasets.monk.MonkDataSource import MonkDataSouce
from functions.loss_functions.LossFunctionFactory import LossFunctionFactory
from validation.grid_search import start_grid_search


class KFoldCV:

    def __init__(self, dataset_name,k, stopping_criteria, possible_hyperparameters, log=False):
        self.means_error = []
        self.k = k
        self.log = log
        self.possible_hyperparameters = possible_hyperparameters
        self.stopping_criteria = stopping_criteria
        self.dataset_name = dataset_name
        if dataset_name in ["monks-1", "monks-2", "monks-3"]:
            self.dataset = MonkDataSouce(namefile_train="./datasets/monk/" + self.dataset_name + ".train",
                                         namefile_test="./datasets/monk/" + self.dataset_name + ".test")
        else:
            self.dataset = CupDataSource(namefile_train="../../datasets/cup/ML-CUP21-TR.csv",
                                         namefile_test="../../datasets/cup/ML-CUP21-TS.csv")

    def split_input_for_different_processes2(self, hyperparameters):
        """
        :param hyperparameters: set of hyperparameters
        :return: a list of dictionaries containing the description of a single fold
        """
        inputs, targets = self.dataset.getDatasetForValidation()
        # Split the dataset into k mutually exclusive subsets D_1,D_2,…,D_k
        input_k_splitted = np.array_split(inputs, self.k)
        target_k_splitted = np.array_split(targets, self.k)

        input_for_processes = list()

        i = 0
        # D_1,D_2,…,D_k will be used as VL sets
        for VL_inputs, VL_targets in zip(input_k_splitted, target_k_splitted):
            # the sets of examples that are not in D_i will be used as TR sets
            TR_inputs = np.concatenate(input_k_splitted[:i] + input_k_splitted[i + 1:])
            TR_targets = np.concatenate(target_k_splitted[:i] + target_k_splitted[i + 1:])
            # every dictionary contains the set of hyperparameters and the structure of a single fold
            inputs_for_the_single_process = {
                "hyperparameters": hyperparameters,
                "k_fold_single_row": {
                    "i": i,
                    "VL_inputs": VL_inputs,
                    "VL_targets": VL_targets,
                    "TR_inputs": TR_inputs,
                    "TR_targets": TR_targets
                }
            }
            i += 1
            input_for_processes.append(inputs_for_the_single_process)

        return input_for_processes

    def compute_mean_error(self, hyperparameters):
        """
        For every set of hyperparametric values
        it computes the mean error over the validation sets obtained by the all folds of each k-fold """
        import time
        start_tot = time.time()

        pool = mp.Pool(mp.cpu_count())
        input_for_processes = self.split_input_for_different_processes2(hyperparameters)
        errors = pool.map(self.async_single_folder, [input for input in input_for_processes])
        pool.close()

        mean = np.sum(errors) / len(errors)
        object_to_save = {
            "mean": mean,
            "hyperparameters": hyperparameters
        }

        time = "Time: " + str(np.round(time.time() - start_tot, 6)) + "sec"

        print("===============")
        print(" *** Model Found ***")
        print(" *** Selected Hyperparameters: ***")
        print(" **" + str(hyperparameters) + "**")
        print(" **  Error:" + str(mean) + "**")
        print(" **  Time:" + str(time) + "**")
        print("===============")
        self.means_error.append(object_to_save)
        return mean

    def async_single_folder(self, input):
        """
        :param input: dictionary containing:
            - set of hyperparameters
            - k_fold_single_row, containing the structure:
            VL_inputs, VL_targets, TR_inputs, TR_targets
        :return: error of the current fold
        """

        k_fold_single_row = input["k_fold_single_row"]
        hyperparameters = input["hyperparameters"]

        VL_inputs = k_fold_single_row["VL_inputs"]
        VL_targets = k_fold_single_row["VL_targets"]
        TR_inputs = k_fold_single_row["TR_inputs"]
        TR_targets = k_fold_single_row["TR_targets"]

        print(">>>\tStart K-Fold: ", k_fold_single_row["i"], "TR = [", len(TR_inputs), "inputs, ", len(TR_targets),
              "targets]", "VL = [", len(VL_inputs), "inputs, ", len(VL_targets), "targets]")

        task = "regression" if self.dataset_name == "cup" else "classification"
        nn = NeuralNetwork(hyperparameters, log=False, task=task)
        nn.train(TR_inputs, TR_targets, VL_inputs, VL_targets)

        # error is estimated over the VL set D_i
        if self.dataset_name == "cup":
            error = nn.estimate_error(VL_inputs, VL_targets, LossFunctionFactory("mee"))
        else:
            error = nn.estimate_error(VL_inputs, VL_targets)

        print("<<<\tEnd K-Fold: ", k_fold_single_row["i"], "| error: ", error)
        return error

    @staticmethod
    def end_of_validation(self, best_hyperparameters):
        """ Retraining of the best model on the whole training dataset
            and estimation of its final error over the test set
        """
        if self.dataset_name != "cup":
            self.__end_monks(best_hyperparameters)
        else:
            self.__end_cup(best_hyperparameters)


    def __end_cup(self, best_hyperparameters):
        inputs, targets = self.dataset.getTrainSet()
        input_testset, targets_testset = self.dataset.getInternalTestSet()
        nn = NeuralNetwork(best_hyperparameters, log=True, forceMetrics=True, task="regression")
        nn.train(inputs, targets, input_testset, targets_testset)
        # Final error on the internal test set
        final_error = nn.estimate_error(input_testset, targets_testset, LossFunctionFactory("mee"))
        print("===============")
        print(" +++ Best Model Found +++")
        print(" +++ Selected Hyperparameters: +++")
        print(" ++" + str(best_hyperparameters) + "++")
        print(" ++  Error:" + str(final_error) + "++")
        print("===============")

    def __end_monks(self, best_hyperparameters):
        inputs, targets = self.dataset.getOriginalTrainingDataset()
        input_testset, targets_testset = self.dataset.getTestSet()
        nn = NeuralNetwork(best_hyperparameters, log=True, forceMetrics=True, task="classification")
        nn.train(inputs, targets, input_testset, targets_testset)
        metrics = nn.get_metrics(input_testset, targets_testset)
        # Final error on the test set
        final_error = nn.estimate_error(input_testset, targets_testset)

        print("===============")
        print(" +++ Best Model Found for " + str(self.dataset_name) + "+++")
        print(" +++ Selected Hyperparameters: +++")
        print(" ++" + str(best_hyperparameters) + "++")
        print(" ++  Error:" + str(final_error) + "++")
        print(" ++  Accuracy = " + str(metrics["accuracy"]) + "++")
        print(" ++  True Positives = " + str(metrics["true_positives"]) + "++")
        print(" ++  True Negatives = " + str(metrics["true_negatives"]) + "++")
        print(" ++  False Positives = " + str(metrics["false_positives"]) + "++")
        print(" ++  False Negatives = " + str(metrics["false_negatives"]) + "++")
        print(" ++  Accuracy = " + str(metrics["accuracy"]) + "++")
        print("===============")

    @staticmethod
    def end_K_Fold(self):
        """
        :return: the best set of hyperparameters, selected by the K-Fold Cross Validation
        """
        means = []
        hyperparameters = []
        for single_folder_result in self.means_error:
            means.append(single_folder_result["mean"])
            hyperparameters.append(single_folder_result["hyperparameters"])
        # choose the set of hyperparameters which gives the minimum mean error
        index_of_the_best_coarse = np.argmin(means)
        self.means_error.clear()
        return hyperparameters[index_of_the_best_coarse]

    def start(self, useFineGS=False):
        """ K-Fold Cross Validation """
        # a first coarse Grid Search, values differ in order of magnitude
        best_hyperparameter_coarse = start_grid_search(context=self,
                                                       hyperparameters=self.possible_hyperparameters,
                                                       isCoarse=True,
                                                       callback_for_each_hyperparameter=KFoldCV.compute_mean_error,
                                                       stopping_criteria=self.stopping_criteria,
                                                       final_callback=KFoldCV.end_K_Fold,
                                                       db_initialization=False)
        """ best set of hyperparameters found """
        result = best_hyperparameter_coarse

        """  a finer Grid Search, values are taken in a small range of the winner values of the Coarse Grid Search """
        if useFineGS:
            result = start_grid_search(context=self,
                                       hyperparameters=result,
                                       isCoarse=False,
                                       callback_for_each_hyperparameter=KFoldCV.compute_mean_error,
                                       stopping_criteria=self.stopping_criteria,
                                       final_callback=KFoldCV.end_K_Fold,
                                       db_initialization=False)

        """ retraining of the best model found """
        KFoldCV.end_of_validation(self, best_hyperparameters=result)


    def start_distributed(self, isCoarse):
        while True:
            import Constants
            print("======= ["+Constants.who+"] Start a folder ==========")
            try:
                """ Taking an attempt not yet calculated from the database """
                attempt = kfold_distributed_computing_cup.getUnsolvedAttempt(isCoarse)
                id = attempt[0]
                hyperparam = attempt[1]
                print("======= ["+Constants.who+"] Hyperparameter set #"+str(id)+"  ==========")

                """ Executing a single Folder with these hyperparams """
                mean = self.compute_mean_error(hyperparam)

                print("======= ["+Constants.who+"] id = "+str(id)+", result = "+str(mean)+"  ==========")
                if math.isnan(mean): # Divergent case
                    mean = -1

                """ At the end, we write the result on the database """
                kfold_distributed_computing_cup.postResult(id, mean, isCoarse)
            except  Exception as e:
                print("======= [" + Constants.who + "] FAIL ==========")
                print(e)
                import time
                time.sleep(5)
                pass