from functions.lr_decay_functions.LearningRateDecay import LearningRateDecay


class LearningRateDecay_No(LearningRateDecay):

    @staticmethod
    def getNextLearningRate(actual_lr, initial_lr, actual_epoch):
        return actual_lr
