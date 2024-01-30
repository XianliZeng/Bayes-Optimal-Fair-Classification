

from joblib import Parallel, delayed
import numpy as np
###Preprocessing###
from Algorithms.algorithm_FUDS import training_FUDS
from Algorithms.algorithm_DIR import training_DIR
from Algorithms.algorithm_FAWOS import training_FAWOS

###Inprocessing###
from Algorithms.algorithm_FCSC import training_FCSC
from Algorithms.algorithm_KDE import training_KDE
from Algorithms.algorithm_ADV import training_ADV

###Postprocessing###
from Algorithms.algorithm_FPIR import training_FPIR
from Algorithms.algorithm_PPF import training_PPF
from Algorithms.algorithm_PPOT import training_PPOT



###Considered datasets###




lambda_list = np.arange(10) / 10 + 0.05
alphaF_list = np.arange(10) / 5
alphaA_list = np.arange(10) / 4 + 0.2
level_list_law = level_list = np.arange(10) / 125
level_list_adult = np.arange(10) / 50
level_list_compas = 0.3 * np.arange(10) / 10


def get_parameter_list(method, dataset):
    if method == 'FUDS':
        if dataset == 'AdultCensus':
            parameter_list = level_list_adult
        elif dataset == 'COMPAS':
            parameter_list = level_list_compas
        elif dataset == 'Lawschool':
            parameter_list = level_list_law

    elif method == 'DIR':
        parameter_list = lambda_list

    elif method == 'FAWOS':
        parameter_list = alphaF_list

    elif method == 'FCSC':
        if dataset == 'AdultCensus':
            parameter_list = level_list_adult
        elif dataset == 'COMPAS':
            parameter_list = level_list_compas
        elif dataset == 'Lawschool':
            parameter_list = level_list_law

    elif method == 'KDE':
        parameter_list = lambda_list

    elif method == 'ADV':
        parameter_list = alphaA_list

    else:
        parameter_list = 'None'
    return parameter_list

def training(method,dataset_list,n_seeds,parallel_core_number):
    if method == 'FUDS':
        for dataset in dataset_list:
            parameter_list = get_parameter_list(method,dataset)
            Parallel(n_jobs=parallel_core_number)(
                delayed(training_FUDS)(dataset, delta, seed) for delta in parameter_list for seed in range(n_seeds))


    if method == 'DIR':
        for dataset in dataset_list:
            parameter_list = get_parameter_list(method,dataset)
            Parallel(n_jobs=parallel_core_number)(
                delayed(training_DIR)(dataset, level, seed) for level in parameter_list for seed in range(n_seeds))

    if method == 'FAWOS':
        for dataset in dataset_list:
            parameter_list = get_parameter_list(method,dataset)
            Parallel(n_jobs=parallel_core_number)(
                delayed(training_FAWOS)(dataset, alpha, seed) for alpha in parameter_list for seed in range(n_seeds))



    if method == 'FCSC':
        for dataset in dataset_list:
            parameter_list = get_parameter_list(method,dataset)
            Parallel(n_jobs=parallel_core_number)(
                delayed(training_FCSC)(dataset, delta, seed) for delta in parameter_list for seed in range(n_seeds))



    if method == 'KDE':
        for dataset in dataset_list:
            parameter_list = get_parameter_list(method,dataset)
            Parallel(n_jobs=parallel_core_number)(
                delayed(training_KDE)(dataset, lambda_, seed) for lambda_ in parameter_list for seed in range(n_seeds))



    if method == 'ADV':
        for dataset in dataset_list:
            parameter_list = get_parameter_list(method,dataset)
            Parallel(n_jobs=parallel_core_number)(
                delayed(training_ADV)(dataset, alpha, seed) for alpha in parameter_list for seed in range(n_seeds))



    if method == 'FPIR':
        Parallel(n_jobs=parallel_core_number)(
                delayed(training_FPIR)(dataset,  seed) for dataset in dataset_list for seed in range(n_seeds))




    if method == 'PPOT':
        Parallel(n_jobs=parallel_core_number)(
                delayed(training_PPOT)(dataset,  seed) for dataset in dataset_list for seed in range(n_seeds))



    if method == 'PPF':
        Parallel(n_jobs=parallel_core_number)(
                delayed(training_PPF)(dataset,  seed) for dataset in dataset_list for seed in range(n_seeds))

