from functions.lr_decay_functions.LearningRateDecay_Exponential import LearningRateDecay_Exponential
from functions.lr_decay_functions.LearningRateDecay_Linear import LearningRateDecay_Linear
from functions.lr_decay_functions.LearningRateDecay_No import LearningRateDecay_No

class LearningRateDecayFactory:
    """ This is the factory for LR decay functions,
            this design pattern allows us to have a more maintainable code. """
    linear = 'linear'
    exponential = 'exponential'
    no = 'no'

    def __init__(self, learningRateDecaySelected):
        if learningRateDecaySelected == 'linear':
            self.theFunction = LearningRateDecay_Linear()
        elif learningRateDecaySelected == 'exponential':
            self.theFunction = LearningRateDecay_Exponential()
        elif learningRateDecaySelected == 'no':
            self.theFunction = LearningRateDecay_No()
        else:
            raise Exception("Invalid value: "+ learningRateDecaySelected+" is not a learning rate decay type")

    def getNextLearningRate(self, actual_lr, initial_lr, actual_epoch):
        newLr = self.theFunction.getNextLearningRate(actual_lr, initial_lr, actual_epoch)
        return newLr
