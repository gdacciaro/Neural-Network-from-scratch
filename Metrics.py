import math

import numpy as np

import Utility
from functions.loss_functions.LossFunctionFactory import LossFunctionFactory

def getMetricsCup(neuralNetwork, inputs, targets):
    mee = 0
    mee_funct = LossFunctionFactory("mee")
    for input, target in zip(inputs, targets):
        output = neuralNetwork.feedforward(input)
        single_mee = mee_funct.getFunction(predicted=output, target=target)
        mee += single_mee
    mee /= len(inputs)
    return mee


def getMetrics(neuralNetwork, inputs, targets, log = False):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for input, target in zip(inputs, targets):
        result = Utility.apply_classification_threshold(neuralNetwork.feedforward(input))

        if np.equal(result, target):
            if np.equal(target, 0):
                true_negatives+=1
            else:
                true_positives+=1
        else:
            if np.equal(target, 0):
                false_negatives+=1
            else:
                false_positives+=1

    #https://towardsdatascience.com/20-popular-machine-learning-metrics-part-1-classification-regression-evaluation-metrics-1ca3e282a2ce
    accuracy = np.round((true_positives+true_negatives)/(true_positives+true_negatives+false_positives+false_negatives)*100,3)
    try:
        precision_pos =  np.round(true_positives / (true_positives + false_positives) * 100, 3)
    except:
        precision_pos = math.nan
    try:
        precision_neg = np.round(true_negatives / (true_negatives + false_negatives) * 100, 3)
    except:
        precision_neg = math.nan

    try:
        recall = np.round(true_positives / (true_positives + false_negatives) * 100, 3)
    except:
        recall = 0

    try:
        f1score = np.round(true_positives / (true_positives + false_negatives) * 100, 3)
    except:
        f1score = 0



    if log :
        print("\n\n=========== Metrics ===========")
        print("True positives\t",true_positives)
        print("True negatives\t",true_negatives)
        print("False positives\t",false_positives)
        print("False negatives\t",false_negatives, "\n")

        m = [
             ["TP "+str(true_positives), "FN "+str(false_negatives)],
             ["FP "+str(false_positives), "TN "+str(true_negatives)]
             ]
        print("Confusion Matrix:\t\t")
        Utility.well_print_matrix(m)
        print("\nOther metrics:")
        print("\tAccuracy\t\t\t",accuracy)
        print("\tPrecision_pos\t\t",precision_pos)
        print("\tPrecision_neg\t\t",precision_neg)
        print("\tRecall\t\t\t\t",recall)
        print("\tF1 score\t\t\t",f1score)

    metrics = {
        "accuracy":accuracy,
        "true_positives":true_positives,
        "true_negatives":true_negatives,
        "false_positives":false_positives,
        "false_negatives":false_negatives,
        "precision_pos":precision_neg,
        "recall":recall,
        "f1score":f1score
    }
    return metrics