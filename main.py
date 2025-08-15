import time
import os

import IPython
import pandas as pd

from training_setting import training, get_parameter_list
from analysis_all import drow_plot, print_table

#dataset_list = ['COMPAS', 'AdultCensus', 'Lawschool', 'ACSIncome']
dataset_list = ['ACSIncome']
model = 'hgb' #mlp, lgbm, hgb

n_seeds = 10 #Number of random seeds to try
IPython.display.clear_output()

#3 proposed + 6 baseline
#methods_to_train = ['FUDS', 'FCSC', 'FPIR', 'DIR', 'FAWOS', 'KDE', 'ADV', 'PPOT', 'PPF']
#methods_to_train = ['FUDS', 'FCSC', 'FPIR', 'DIR', 'KDE', 'ADV', 'PPOT', 'PPF'] #No FAWOS for ACSIncome

#3 proposed
#methods_to_train = ['FUDS', 'FCSC', 'FPIR']
methods_to_train = ['FCSC']

# 6 baseline
#methods_to_train = ['DIR', 'FAWOS', 'KDE', 'ADV', 'PPF', 'PPOT']
#methods_to_train = ['ADV']

# additional baseline
#methods_to_train = ['MinDiff', 'FRAPPE', 'RED', 'ARA']
#methods_to_train = ['RED']

#Custom
#methods_to_train = ['ADV', 'PPF', 'PPOT', 'FAWOS', 'KDE'] #OTHER ; OMIT FAWOS for ACSIncome

#ALL 
#methods_to_train = ['FUDS', 'FCSC', 'FPIR', 'DIR', 'FAWOS', 'KDE', 'ADV', 'PPOT', 'PPF', 'MinDiff', 'FRAPPE', 'ARA'] #Except RED

############training##########
parallel_core_number = int(os.environ['NSLOTS'])
for method in methods_to_train:
    training(method,dataset_list = dataset_list,n_seeds=n_seeds,parallel_core_number=parallel_core_number,model=model)

############merge results########
# default + per-method overrides, as before
n_seeds_dict = {
    'REDd': 10
}

for dataset in dataset_list:
    for method in methods_to_train:
        seeds_for_method = n_seeds_dict.get(method, n_seeds)
        Result_all = []

        # helper to find the right file
        def load_csv(base_path):
            for path in (base_path, base_path + '.csv'):
                if os.path.isfile(path):
                    return pd.read_csv(path)
            raise FileNotFoundError(
                f"Could not find either '{base_path}' nor '{base_path}.csv'"
            )

        if method in ['FPIR', 'PPF', 'FRAPPE', 'PPOT', 'LPP']:
            for seed in range(seeds_for_method):
                base = f'Result/{method}/NNo/result_of_{dataset}_with_seed_{seed}'
                temp = load_csv(base)
                Result_all.append(temp)
        else:
            parameter_list = get_parameter_list(method, dataset)
            for seed in range(seeds_for_method):
                for para in parameter_list:
                    base = (
                        f'Result/{method}/{model.upper()}/result_of_{dataset}'
                        f'_with_seed_{seed}_para_{int(para * 1000)}'
                    )
                    temp = load_csv(base)
                    Result_all.append(temp)

        # concatenate and write out once
        all_df = pd.concat(Result_all, ignore_index=True)
        out_path = (
            f'Result/Result_after_merge/{model.upper()}/All_result_of_{dataset}'
            f'_training_by_{method}.csv'
        )
        all_df.to_csv(out_path, index=False)

###Print tables#####
#print_table(proposed_methods, dataset_list)


###plot_results######
#drow_plot(dataset_list)