from joblib import Parallel, delayed
import numpy as np
###Preprocessing###
from Algorithms.algorithm_FUDS import training_FUDS
from Algorithms.algorithm_DIR import training_DIR
from Algorithms.algorithm_FAWOS import training_FAWOS
from Algorithms.algorithm_ARA import training_ARA

###Inprocessing###
from Algorithms.algorithm_FCSC import training_FCSC
from Algorithms.algorithm_KDE import training_KDE
from Algorithms.algorithm_ADV import training_ADV
from Algorithms.algorithm_RED import training_RED
from Algorithms.algorithm_FRAPPE import training_FRAPPE #in_processing=True

###Postprocessing###
from Algorithms.algorithm_FPIR import training_FPIR
from Algorithms.algorithm_PPF import training_PPF
from Algorithms.algorithm_PPOT import training_PPOT
from Algorithms.algorithm_FRAPPE import training_FRAPPE #in_processing=False
from Algorithms.algorithm_LPP import training_LPP

###Baseline###
from Algorithms.algorithm_BASE import training_BASE

###Considered datasets###
lambda_list = np.arange(10) / 10 + 0.05
alphaF_list = np.arange(10) / 5
alphaA_list = np.arange(10) / 4 + 0.2

ara_list_law = np.linspace(0.00, 0.50, 10) #ARA
ara_list_adult = np.linspace(0.00, 0.50, 10)
ara_list_compas = np.linspace(0.40, 0.50, 10)
ara_list_acsincome = np.linspace(0.00, 0.50, 10)

lambda_mmd_list = [0.0, 0.1, 0.25, 0.50, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0] #MinDiff

level_list_law = level_list = np.arange(10) / 125
level_list_adult = np.arange(10) / 50
level_list_compas = 0.3 * np.arange(10) / 10 * 1
level_list_acsincome = 0.025 * np.arange(10)

blind = False

def get_parameter_list(method, dataset):
    if method == 'FUDS':
        if dataset == 'AdultCensus':
            parameter_list = level_list_adult
        elif dataset == 'COMPAS':
            parameter_list = level_list_compas
        elif dataset == 'Lawschool':
            parameter_list = level_list_law
        elif dataset == 'ACSIncome':
            parameter_list = level_list_acsincome

    elif method == 'DIR':
        parameter_list = lambda_list

    elif method == 'FAWOS':
        parameter_list = alphaF_list
        
    elif method == 'ARA':
        if dataset == 'AdultCensus':
            parameter_list = ara_list_adult
        elif dataset == 'COMPAS':
            parameter_list = ara_list_compas
        elif dataset == 'Lawschool':
            parameter_list = ara_list_law
        elif dataset == 'ACSIncome':
            parameter_list = ara_list_acsincome
  
    elif method == 'FCSC':
        if dataset == 'AdultCensus':
            parameter_list = level_list_adult
        elif dataset == 'COMPAS':
            parameter_list = level_list_compas
        elif dataset == 'Lawschool':
            parameter_list = level_list_law
        elif dataset == 'ACSIncome':
            parameter_list = level_list_acsincome

    elif method == 'KDE':
        parameter_list = lambda_list

    elif method == 'ADV':
        parameter_list = alphaA_list
    
    elif method == 'MinDiff':
        parameter_list = lambda_mmd_list
        
    elif method == 'RED':
        if dataset == 'AdultCensus':
            parameter_list = level_list_adult/2 + 1e-3
        elif dataset == 'COMPAS':
            parameter_list = level_list_compas/2 + 1e-3 #debugging
        elif dataset == 'Lawschool':
            parameter_list = level_list_law/2 + 1e-3
        elif dataset == 'ACSIncome':
            parameter_list = level_list_acsincome/2 + 1e-3

    else:
        parameter_list = 'None'
    return parameter_list

def training(method,dataset_list,n_seeds,parallel_core_number,model="mlp"):
    if method == 'FUDS':
        for dataset in dataset_list:
            parameter_list = get_parameter_list(method,dataset)
            Parallel(n_jobs=parallel_core_number, backend='threading')(
                delayed(training_FUDS)(dataset, delta, False, seed) for delta in parameter_list for seed in range(n_seeds))

    if method == 'DIR':
        for dataset in dataset_list:
            parameter_list = get_parameter_list(method,dataset)
            Parallel(n_jobs=parallel_core_number, backend='threading')(
                delayed(training_DIR)(dataset, level, False, seed) for level in parameter_list for seed in range(n_seeds))

    if method == 'FAWOS':
        for dataset in dataset_list:
            parameter_list = get_parameter_list(method,dataset)
            Parallel(n_jobs=parallel_core_number, backend='threading')(
                delayed(training_FAWOS)(dataset, alpha, False, seed) for alpha in parameter_list for seed in range(n_seeds))
                
    if method == 'ARA':
        for dataset in dataset_list:
            parameter_list = get_parameter_list(method,dataset)
            Parallel(n_jobs=parallel_core_number, backend='threading')(
                delayed(training_ARA)(dataset, alpha, seed) for alpha in parameter_list for seed in range(n_seeds))

    if method == 'FCSC':
        for dataset in dataset_list:
            parameter_list = get_parameter_list(method,dataset)
            Parallel(n_jobs=parallel_core_number, backend='threading')(
                delayed(training_FCSC)(dataset, delta, False, seed, model) for delta in parameter_list for seed in range(n_seeds))

    if method == 'KDE':
        for dataset in dataset_list:
            parameter_list = get_parameter_list(method,dataset)
            Parallel(n_jobs=parallel_core_number, backend='threading')(
                delayed(training_KDE)(dataset, lambda_, False,seed) for lambda_ in parameter_list for seed in range(n_seeds))

    if method == 'ADV':
        for dataset in dataset_list:
            parameter_list = get_parameter_list(method,dataset)
            Parallel(n_jobs=parallel_core_number, backend='threading')(
                delayed(training_ADV)(dataset, alpha, False, seed) for alpha in parameter_list for seed in range(n_seeds))

    if method == 'MinDiff':
        for dataset in dataset_list:
            parameter_list = get_parameter_list(method,dataset)
            Parallel(n_jobs=parallel_core_number, backend='threading')(
                delayed(training_FRAPPE)(dataset, alpha, seed, in_processing=True) for alpha in parameter_list for seed in range(n_seeds))
    
    if method == 'RED':
        max_iter = 50
        for dataset in dataset_list:
            parameter_list = get_parameter_list(method,dataset)
            Parallel(n_jobs=parallel_core_number, backend='threading')(
                delayed(training_RED)(dataset, epsilon, max_iter, seed, model) for epsilon in parameter_list for seed in range(n_seeds))

    if method == 'FPIR':
        Parallel(n_jobs=parallel_core_number, backend='threading')(
                delayed(training_FPIR)(dataset, False, seed) for dataset in dataset_list for seed in range(n_seeds))

    if method == 'PPOT':
        Parallel(n_jobs=parallel_core_number, backend='threading')(
                delayed(training_PPOT)(dataset, seed) for dataset in dataset_list for seed in range(n_seeds))
        
    if method == 'PPF':
        Parallel(n_jobs=parallel_core_number, backend='threading')(
                delayed(training_PPF)(dataset, False, seed) for dataset in dataset_list for seed in range(n_seeds))
                
    if method == 'FRAPPE':
        Parallel(n_jobs=parallel_core_number, backend='threading')(
                delayed(training_FRAPPE)(dataset, None, seed, in_processing=False) for dataset in dataset_list for seed in range(n_seeds))
                
    if method == 'LPP':
        Parallel(n_jobs=parallel_core_number, backend='threading')(
                delayed(training_LPP)(dataset, True, seed) for dataset in dataset_list for seed in range(n_seeds))
        
    if method == 'BASE':
        Parallel(n_jobs=parallel_core_number, backend='threading')(
                delayed(training_BASE)(dataset, False, seed) for dataset in dataset_list for seed in range(n_seeds))

