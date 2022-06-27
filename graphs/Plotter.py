import matplotlib.pyplot as plt

class Plotter:
    """ This is the class used to store and draw graphs """

    def __init__(self):
        self.training_losses = []
        self.testing_losses = []
        self.lr_list = []
        self.mee_train_list = []
        self.mee_test_list = []
        self.training_accuracy_list = []
        self.test_accuracy_list = []
        self.blind_targets_list = []

    def addTrainingLoss(self, loss):
        self.training_losses.append(loss)
    def addTestingLoss(self, loss):
        self.testing_losses.append(loss)
    def addTrainingAccuracy(self, accuracy):
        self.training_accuracy_list.append(accuracy)
    def addTestAccuracy(self, accuracy):
        self.test_accuracy_list.append(accuracy)
    def addMeeTest(self, mee):
        self.mee_test_list.append(mee)
    def addMeeTrain(self, mee):
        self.mee_train_list.append(mee)
    def addLr(self, accuracy):
        self.lr_list.append(accuracy)
    def addBlindTarget(self, target_0, target_1):
        self.blind_targets_list.append((target_0, target_1))

    def plot(self, list):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        index_of_losses = -1
        index_of_accuracy = -1
        index_of_mee = -1
        for index,item in enumerate(list):
            if item == "lr":
                title = "Learning Rates"
                axs[index].plot(self.lr_list)
                axs[index].set_title(title)
                axs[index].set_xlabel('Epochs')

            elif item == "training_accuracy":
                if index_of_accuracy == -1:
                    index_of_accuracy = index
                title = "Accuracy"
                axs[index_of_accuracy].plot(range(len(self.training_accuracy_list)), self.training_accuracy_list, label="Train", linestyle='dashed', color='b')
                axs[index_of_accuracy].set_title(title)
                axs[index_of_accuracy].legend(loc="best", prop={'size': 15})
                axs[index_of_accuracy].grid()
                axs[index_of_accuracy].set_xlabel('Epochs')

            elif item == "test_accuracy":
                if index_of_accuracy == -1:
                    index_of_accuracy = index
                title = "Accuracy"
                axs[index_of_accuracy].plot(range(len(self.test_accuracy_list)), self.test_accuracy_list, color='r', label="Test")
                axs[index_of_accuracy].set_title(title)
                axs[index_of_accuracy].grid()
                axs[index_of_accuracy].legend(loc="best", prop={'size': 15})
                axs[index_of_accuracy].set_xlabel('Epochs')

            elif item == "training_loss":
                if index_of_losses == -1:
                    index_of_losses = index
                title = "Loss (MEE)"
                axs[index_of_losses].plot(range(len(self.training_losses)), self.training_losses, color='b', label="Train", linestyle='dashed')
                axs[index_of_losses].set_title(title)
                axs[index_of_losses].legend(loc="best", prop={'size': 15})
                axs[index_of_losses].grid()
                axs[index_of_losses].set_xlabel('Epochs')

            elif item == "test_loss":
                if index_of_losses == -1:
                    index_of_losses = index
                title = "Loss (MEE)"
                axs[index_of_losses].plot(range(len(self.testing_losses)), self.testing_losses, color='r', label="Test")
                axs[index_of_losses].set_title(title)
                axs[index_of_losses].semilogy()
                axs[index_of_losses].grid()
                axs[index_of_losses].legend(loc="best", prop={'size': 15})
                axs[index_of_losses].set_xlabel('Epochs')

            elif item == "blind_targets":
                title = "Blind Targets"
                for targets in self.blind_targets_list:
                    axs[index].scatter(targets[0], targets[1])
                axs[index].set_title(title)
                axs[index].set_xlabel('Epochs')

            elif item == "test_mee":
                if index_of_mee == -1:
                    index_of_mee = index
                title = "Losses"
                axs[index_of_mee].plot(range(len(self.mee_test_list)), self.mee_test_list, color='green', linestyle='dashed',
                                          label="MEE - Test")
                axs[index_of_mee].set_title(title)
                axs[index_of_mee].semilogy()
                axs[index_of_mee].grid()
                axs[index_of_mee].legend(loc="best", prop={'size': 15})
                axs[index_of_mee].set_xlabel('Epochs')

            elif item == "train_mee":
                if index_of_mee == -1:
                    index_of_mee = index
                title = "Losses"
                axs[index_of_mee].plot(range(len(self.mee_train_list)), self.mee_train_list, color='purple',
                                          label="MEE -  Train")
                axs[index_of_mee].set_title(title)
                axs[index_of_mee].semilogy()
                axs[index_of_mee].grid()
                axs[index_of_mee].legend(loc="best", prop={'size': 15})
                axs[index_of_mee].set_xlabel('Epochs')
        fileName = 'plot.png'
        plt.savefig(fileName)
        fig.show()

    def plot_blind_target(self,):
        title = "Blind Targets"
        for targets in self.blind_targets_list:
            plt.scatter(targets[0], targets[1])
        plt.title(title)
        plt.show()