from mysql.connector import (connection)

import Constants
from Layer import Layer
from validation.grid_search import start_grid_search
from Hyperparameters import Hyperparameters

def writeTemp(hyperparamas, isCoarse):
    connection_write = connection.MySQLConnection(user=Constants.MySQL_user, password=Constants.MySQL_password,
                                                  host=Constants.MySQL_host, database=Constants.MySQL_db)
    nameTable = __getTableName(isCoarse)

    cursor = connection_write.cursor()
    query = ("INSERT INTO "+nameTable +  " (reserved, alpha_momentum, lr, L2, batch_size, topology, lr_decay_type, result, who) "
                    "VALUES ( 0, "+str(hyperparamas.alpha_momentum)+",  "+str(hyperparamas.start_lr)+",  "+str(hyperparamas.L2)+", \""
                            +str(hyperparamas.batch_size)+"\",  "+
                            " \""+str(hyperparamas.topology)+"\", \""+str(hyperparamas.lr_decay_type)+"\", NULL, \""+""+"\")")

    print(query)
    cursor.execute(query)
    connection_write.commit()
    cursor.close()

def init_database(isCoarse):
    if not isCoarse:

        """ These hyperparameters are the winner of the Coarse GS for the CUP's dataset """
        topology = [
            Layer(10, activation='none'),
            Layer(32, activation='sigmoid'),
            Layer(20, activation='sigmoid'),
            Layer(2, activation='identity'),
        ]

        start_lr = 0.1
        L2 = 0.00001
        alpha_momentum = 0.5
        batch_size = 32
        lr_decay_type = "linear"

        threshold_variance = Constants.stopping_criteria["threshold_variance"]
        max_epochs = Constants.stopping_criteria["max_epoch"]

        best_hyperparam = Hyperparameters(topology=topology, start_lr=start_lr, L2=L2,
                                          alpha_momentum=alpha_momentum, batch_size=batch_size,
                                          max_epochs=max_epochs, nesterov=False,
                                          threshold_variance=threshold_variance,
                                          lr_decay_type=lr_decay_type)

        start_grid_search(db_initialization=True, isCoarse=isCoarse, stopping_criteria=Constants.stopping_criteria,
                          context=None, hyperparameters=best_hyperparam,
                          callback_for_each_hyperparameter=None, final_callback=None)
    else:

        default_hyper = Hyperparameters.get_default_hyperparameters("cup") #We used the distributed calculus only for CUP

        start_grid_search(db_initialization=True, isCoarse=isCoarse, stopping_criteria=Constants.stopping_criteria,
                          context=None, hyperparameters=default_hyper,
                          callback_for_each_hyperparameter=None, final_callback=None)



def getUnsolvedAttempt(isCoarse):
    """ This function returns an attempt that is not yet reserved """
    try:
        cnx = connection.MySQLConnection(user=Constants.MySQL_user, password=Constants.MySQL_password,
                                         host=Constants.MySQL_host, database=Constants.MySQL_db)
        cursor = cnx.cursor()
    except:
        return

    """ We obtain the first non-reserved set of hyperparameters ... """
    nameTable = __getTableName(isCoarse)
    query = "select * from "+nameTable+" where reserved = 0 LIMIT 1"
    cursor.execute(query)

    """ .. we parse it .. """
    hyperparam = None
    for (id, reserved, alpha_momentum, lr, L2, batch_size, topology, lr_decay_type, result, who ) in cursor:
        max_epochs = Constants.stopping_criteria["max_epoch"]
        threshold_variance =  Constants.stopping_criteria["threshold_variance"]
        realTopology = __parseTopology(topology)

        hyperparam = Hyperparameters(alpha_momentum= alpha_momentum,
                                     start_lr=lr,
                                     L2=L2,
                                     batch_size= batch_size,
                                     nesterov = False,
                                     lr_decay_type=lr_decay_type,
                                     max_epochs=max_epochs,
                                     topology=realTopology,
                                     threshold_variance =threshold_variance)

    """ ... and then, we reserve it to our worker """
    try:
        worker_name = Constants.who
        query2 = "update "+nameTable+" set reserved=1, who=\""+worker_name+"\" where id="+str(id)
        cursor.execute(query2)
        cnx.commit()
        cursor.close()
        cnx.close()
    except:
        return

    return id, hyperparam

def postResult(id, mean, isCoarse):
    """ Once we calculated our attempt, we can post the result on the database"""
    cnx = connection.MySQLConnection(user=Constants.MySQL_user, password=Constants.MySQL_password,
                                     host=Constants.MySQL_host, database=Constants.MySQL_db)
    cursor = cnx.cursor()

    nameTable = __getTableName(isCoarse)

    query2 = "update "+nameTable+" set result="+str(mean)+" where id=" + str(id)
    cursor.execute(query2)
    cnx.commit()
    cursor.close()
    cnx.close()


def __parseTopology(topology):
    topology = topology.replace("[", "")
    topology = topology.replace("]", "")
    topology = topology.replace("n:", "")
    topology = topology.replace("act=", "")
    topology = topology.replace(" ", "")
    splitted = topology.split(",")
    finalTopology = list()
    for i in range(0, len(splitted), 2):
        neurons = splitted[i]
        fact = splitted[i+1]
        layer = Layer(nodes=neurons, activation=fact)
        finalTopology.append(layer)
    return finalTopology


def __getTableName(isCoarse):
    if isCoarse:
        nameTable = Constants.MySQL_name_table_coarseGS
    else:
        nameTable = Constants.MySQL_name_table_fineGS
    return nameTable