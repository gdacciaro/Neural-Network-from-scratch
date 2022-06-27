from functions.lr_decay_functions.LearningRateDecay import LearningRateDecay


class LearningRateDecay_Linear(LearningRateDecay):

    @staticmethod
    def getNextLearningRate(actual_lr, initial_lr, actual_epoch):
        tau_step = 250
        min_lr = 0.01*initial_lr

        if actual_epoch < tau_step and actual_lr > min_lr:
            alpha = actual_epoch / tau_step
            curr_lr = (1. - alpha) * initial_lr + alpha * min_lr
            return curr_lr

        return min_lr