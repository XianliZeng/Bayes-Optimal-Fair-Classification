import time
import os

import IPython
import pandas as pd

from training_setting import training, get_parameter_list
from analysis_all import draw_plot, print_table

dataset_list = ['COMPAS', 'AdultCensus', 'Lawschool', 'ACSIncome']
model = 'mlp' #mlp, lgbm, hgb #Model to use for pre-training classifier

n_seeds = 50 #Number of random seeds to try
IPython.display.clear_output()

#3 proposed
#methods_to_train = ['FUDS', 'FCSC', 'FPIR']

#10 baseline
methods_to_train = ['DIR', 'FAWOS', 'ARA'] #pre-processing
methods_to_train = ['KDE', 'ADV', 'MinDiff', 'RED'] #in-processing
methods_to_train = ['PPF', 'PPOT', 'FRAPPE'] #post-processing

############training##########
parallel_core_number = int(os.environ['NSLOTS'])
for method in methods_to_train:
    training(method,dataset_list = dataset_list,n_seeds=n_seeds,parallel_core_number=parallel_core_number,model=model)

############merge results########
for dataset in dataset_list:
    for method in methods_to_train:
        seeds_for_method = n_seeds
        Result_all = []

        # helper to find the right file
        def load_csv(base_path):
            for path in (base_path, base_path + '.csv'):
                if os.path.isfile(path):
                    return pd.read_csv(path)
            raise FileNotFoundError(
                f"Could not find either '{base_path}' nor '{base_path}.csv'"
            )

        if method in ['FPIR', 'PPF', 'PPOT', 'FRAPPE']:
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
print_table(proposed_methods, dataset_list)
###plot_results######
draw_plot(dataset_list)
