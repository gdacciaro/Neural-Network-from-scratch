import numpy as np

import Constants
import time
from validation.KFoldCV import KFoldCV

if __name__ == '__main__':
    """ 
    Since Grid Search calculation for the CUP is computationally expensive, we thought of distributing it among computers, using a centralised database to store the results.
    The final retraining was then performed manually in the manual_retraining.py file.
    """

    dataset_name = "cup"

    Constants.who = "worker-"+
    isCoarse = True

    #Execute the database's initialization only once:
    #init_database(isCoarse)

    start_tot = time.time()
    KFoldCV(dataset_name=dataset_name, k=4, possible_hyperparameters=None, stopping_criteria=Constants.stopping_criteria)\
                 .start_distributed(isCoarse)
    time = "K-Fold Time: " + str(np.round(time.time() - start_tot, 6)) + "sec"
    print(time)
