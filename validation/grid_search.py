from validation.distribuited_computing import kfold_distributed_computing_cup
from Hyperparameters import Hyperparameters


def populate_value_for_grid_search(hyperparameters, isCoarse):
    """ This function fills the set of possibile hyperparameters for the Grid Search
        in two possible ways, for a coarse grid search and for a finer one
    """
    global possibles_lr, possibles_alpha_momentum, possibles_L2, possibles_batch_size, possibles_lr_decay_type, possibles_topologies

    # Default values for Coarse Grid Search (values differ in order of magnitude)
    if isCoarse:
        possibles_lr             = hyperparameters["lr"]
        possibles_alpha_momentum = hyperparameters["alpha_momentum"]
        possibles_L2             = hyperparameters["L2"]
        possibles_batch_size     = hyperparameters["batch_size"]
        possibles_lr_decay_type  = hyperparameters["lr_decay_type"]
        possibles_topologies     = hyperparameters["topologies"]

    # Edited values for Fine Grid, taken in a small range of the winner values of the Coarse Grid Search
    else:
        possibles_batch_size     = Hyperparameters.get_fine_param_batch_size(hyperparameters.batch_size)

        possibles_lr             = Hyperparameters.get_fine_param(hyperparameters.start_lr)
        possibles_alpha_momentum = Hyperparameters.get_fine_param(hyperparameters.alpha_momentum)
        possibles_L2             = Hyperparameters.get_fine_param(hyperparameters.L2)

        possibles_topologies     = [hyperparameters.topology]
        possibles_lr_decay_type  = [hyperparameters.lr_decay_type]


def start_grid_search(context, hyperparameters, isCoarse, callback_for_each_hyperparameter, stopping_criteria, final_callback, db_initialization):
    """ Grid Search
        :param : context, a reference of the KFold class
        :param : hyperparameters, in the case of Coarse GS, indicates the list of possible hyperparameters to be searched.
                                  in the case of Fine GS, indicates the best hyperparameter (which won the Coarse GS)
        :param : isCoarse, a Boolean value
        :param : callback_for_each_hyperparameter, a callback which is called in order to calculate its folder
        :param : final_callback, a callback which is called at the end of the GS
        :param : stopping_criteria, an object for stopping criteria
        :param : db_initialization, a Boolean value for inizialize for the centralized database
    """

    populate_value_for_grid_search(hyperparameters, isCoarse)

    max_epochs = stopping_criteria["max_epoch"]
    threshold_variance = stopping_criteria["threshold_variance"]

    index = 0
    sum = len(possibles_lr) * len(possibles_alpha_momentum) * len(possibles_L2) \
          * len(possibles_batch_size) \
          * len(possibles_topologies) \
          * len(possibles_lr_decay_type)

    """ cycle over all the possible values of hyperparameters """
    for lr in possibles_lr:
        for alpha in possibles_alpha_momentum:
            for L2 in possibles_L2:
                for batch_size in possibles_batch_size:
                        for topology in possibles_topologies:
                            for lr_decay_type in possibles_lr_decay_type:
                                print("Trying the hyperparam set :",index,"/", sum)
                                index += 1

                                hyperparameters = Hyperparameters(topology=topology, start_lr=lr, L2=L2,
                                                                  alpha_momentum=alpha,
                                                                  batch_size=batch_size,
                                                                  lr_decay_type=lr_decay_type,
                                                                  max_epochs=max_epochs,
                                                                  threshold_variance=threshold_variance,
                                                                  nesterov=False)

                                #Centralized database initialization
                                if db_initialization:
                                    kfold_distributed_computing_cup.writeTemp(hyperparameters, isCoarse)
                                    continue

                                # for each set of hyperparameters calculate the mean error of K-Fold CV ...
                                callback_for_each_hyperparameter(context, hyperparameters)

    #... and return the best set, i.e. the one with the lowest mean error
    if not db_initialization:
        return final_callback(context)
