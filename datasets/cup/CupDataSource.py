""" Questa classe rappresenta """
import Constants


class CupDataSource:

    def __init__(self, namefile_test, namefile_train):
        columns = ["id", "input_0", "input_1", "input_2", "input_3", "input_4", "input_5", "input_6", "input_7",
                   "input_8", "input_9", "target_0", "target_1"]
        self.namefile_train = namefile_train
        self.namefile_test = namefile_test
        self.original_input_train, self.original_target_train = self.__read_file(namefile_train, columns)
        self.input_test, _ = self.__read_file(namefile_test, columns)

        threshold_split_TR_and_TS = .70  # 70% TR -> 30% TS

        self.input_TR = self.original_input_train[: int(len(self.original_input_train) * threshold_split_TR_and_TS)]
        self.target_TR = self.original_target_train[: int(len(self.original_target_train) * threshold_split_TR_and_TS)]
        self.input_TS = self.original_input_train[int(len(self.original_input_train) * threshold_split_TR_and_TS):]
        self.target_TS = self.original_target_train[int(len(self.original_target_train) * threshold_split_TR_and_TS):]

    def __read_file(self, namefile, columns):
        import pandas as pd

        cup = pd.read_csv(namefile, sep=",", skiprows=[0, 1, 2, 3, 4, 5, 6], header=None, names=columns)
        cup.set_index('id', inplace=True)

        target_0 = cup.pop('target_0')
        target_1 = cup.pop('target_1')

        targets = list()
        for a, b in list(zip(target_0, target_1)):
            targets.append([a, b])

        return cup.values, targets

    def getTrainSet(self):
        return self.input_TR, self.target_TR

    def getInternalTestSet(self):
        return self.input_TS, self.target_TS

    def getTRAndVL_forRetraining(self):
        threshold_split = .85  # 85% TR -> 15% VL

        input_TR_ret    = self.input_TR[: int(len(self.input_TR) * threshold_split)]
        target_TR_ret = self.target_TR[: int(len(self.target_TR) * threshold_split)]
        input_VL_ret   = self.input_TR[int(len(self.input_TR) * threshold_split):]
        target_VL_ret  = self.target_TR[int(len(self.target_TR) * threshold_split):]

        return input_TR_ret, target_TR_ret, input_VL_ret, target_VL_ret


    def getBlindTestSet(self):
        return self.input_test

    def getOriginalTrainingDataset(self):
        return self.original_input_train, self.original_target_train

    def getDatasetForValidation(self):
        return self.getTrainSet()

    def plotTargets(self):
        import matplotlib.pyplot as plt
        x = [item[0] for item in self.original_target_train]
        y = [item[1] for item in self.original_target_train]
        plt.scatter(x, y, label='Data')
        plt.xlabel('target[0]')
        plt.ylabel('target[1]')
        plt.legend()
        plt.show()

    @staticmethod
    def write_results_on_csv(targets):  # a list of numpy 2 elements array ordered as the input_test
        import pandas as pd
        fileName = './'+Constants.team_name+'_ML-CUP21-TS.csv'

        cup = pd.DataFrame()
        target_0 = list()
        target_1 = list()

        for t in targets:
            target_0.append(t[0])
            target_1.append(t[1])

        cup.insert(len(cup.columns), "target_0", target_0)
        cup.insert(len(cup.columns), "target_1", target_1)
        header = ['# Gennaro Daniele Acciaro,  Annachiara Aiello,  Alessandro Bucci',
                  '# '+Constants.team_name,
                  '# ML-CUP21',
                  '# Submission Date (e.g. 04/01/2022)']
        textfile = open(fileName, "w")
        for element in header:
            textfile.write(element + "\n")
        textfile.close()

        cup.to_csv(fileName,  mode='a', header=False, index=True)

