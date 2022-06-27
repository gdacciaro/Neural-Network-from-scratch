import time
import numpy as np

import Constants
from Hyperparameters import Hyperparameters
from validation.KFoldCV import KFoldCV

if __name__ == '__main__':
    dataset_name = "cup"

    # set of hyperparametric values used for the Grid Search
    possible_hyperparameters = Hyperparameters.get_default_hyperparameters(dataset_name)


    start_tot = time.time()
    KFoldCV(dataset_name=dataset_name, k=4,
            possible_hyperparameters=possible_hyperparameters,
            stopping_criteria=Constants.stopping_criteria).start()
    time = "K-Fold Time: " + str(np.round(time.time() - start_tot, 6)) + "sec"
    print(time)
