import time
import numpy as np

from Hyperparameters import Hyperparameters
from validation.KFoldCV import KFoldCV

if __name__ == '__main__':
    dataset_name = "monks-1"
    #dataset_name = "monks-2"
    #dataset_name = "monks-3"
    #dataset_name = "monks-3-reg"

    # set of hyperparametric values used for the Grid Search
    possible_hyperparameters = Hyperparameters.get_default_hyperparameters(dataset_name)

    stopping_criteria = {
        "max_epoch": 150,
        "threshold_variance": 0.
    }

    if dataset_name.startswith("monks-3"):
        dataset_name = "monks-3"

    start_tot = time.time()
    KFoldCV(dataset_name=dataset_name, k=6,
            possible_hyperparameters=possible_hyperparameters,
            stopping_criteria=stopping_criteria).start()
    time = "K-Fold Time: " + str(np.round(time.time() - start_tot, 6)) + "sec"
    print(time)
