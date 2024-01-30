import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_HALF_UP


def print_table(methods, dataset_list):
    list = [0, 2, 4, 6 ,8]
    for dataset in dataset_list:
        if dataset == 'AdultCensus':
            print('\\multicolumn{6}{c}{AdultCensus}  \\\\\\hline')
            print('Methods & $\delta$ & 0.00 & 0.04 &0.08 & 0.12 & 0.16', end='\\\\\\hline\n')
        if dataset == 'COMPAS':
            print('\\multicolumn{6}{c}{COMPAS}  \\\\\\hline')
            print('Methods & $\delta$ & 0.00 & 0.06 &0.12 & 0.18 & 0.24', end='\\\\\\hline\n')

        if dataset == 'Lawschool':
            print('\\multicolumn{6}{c}{Lawschool}  \\\\\\hline')
            print('Methods & $\delta$ & 0.000 & 0.0016 &0.032 & 0.48 & 0.64', end='\\\\\\hline\n')
        for method in methods:
            result_FUDS = pd.read_csv(f'Result/Result_after_merge/All_result_of_{dataset}_training_by_{method}')
            ddp_mean_list = [result_FUDS[t::10]['disparity'].mean() for t in range(10)]
            ddp_std_list = [result_FUDS[t::10]['disparity'].std() for t in range(10)]
            if method == 'FUDS':
                print('FUDS &    & ', end=' ')
            if method == 'FCSC':
                print('FCSC & DD & ', end=' ')
            if method == 'FPIR':
                print('FPIR &    & ', end=' ')
            for l in list:
                ddp_mean = Decimal(ddp_mean_list[l]).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
                print(f'{ddp_mean}', end=' ')
                ddp_std = Decimal(ddp_std_list[l]).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
                if l == 8:
                    if method == 'FPIR':
                        print(f'({ddp_std})', end=' \\\\\\hline\n ')
                    else:
                        print(f'({ddp_std})', end=' \\\\\n ')
                else:
                    print(f'({ddp_std})', end=' & ')


def drow_plot(dataset_list):
    for dataset in dataset_list:
        if dataset == 'AdultCensus':
            xticks_pre = [0.825, 0.833, 0.841, 0.849]
            xticks_in = [0.829, 0.836, 0.843, 0.850]
            xticks_post = [0.834, 0.839, 0.844, 0.849]
            yticks = [0.03, 0.07, 0.11, 0.15]

        if dataset == 'COMPAS':
            xticks_pre = [0.632, 0.646, 0.66, 0.674]
            xticks_in = [0.663, 0.667, 0.671, 0.675]
            xticks_post = [0.664, 0.668, 0.672, 0.676]
            yticks = [0.05, 0.11, 0.17, 0.23]

        if dataset == 'Lawschool':
            xticks_pre = [0.7855, 0.7868, 0.7881, 0.7894]
            xticks_in = [0.787, 0.7878, 0.7886, 0.7894]
            xticks_post = [0.7874, 0.7881, 0.7888, 0.7895]
            yticks = [0.01, 0.03, 0.05, 0.07]

        result_FUDS = pd.read_csv(f'Result/Result_after_merge/All_result_of_{dataset}_training_by_FUDS')
        result_FAWOS = pd.read_csv(f'Result/Result_after_merge/All_result_of_{dataset}_training_by_FAWOS')
        result_DIR = pd.read_csv(f'Result/Result_after_merge/All_result_of_{dataset}_training_by_DIR')

        result_KDE = pd.read_csv(f'Result/Result_after_merge/All_result_of_{dataset}_training_by_KDE')
        result_ADV = pd.read_csv(f'Result/Result_after_merge/All_result_of_{dataset}_training_by_ADV')
        result_FCSC = pd.read_csv(f'Result/Result_after_merge/All_result_of_{dataset}_training_by_FCSC')

        result_PPOT = pd.read_csv(f'Result/Result_after_merge/All_result_of_{dataset}_training_by_PPOT')
        result_PPF = pd.read_csv(f'Result/Result_after_merge/All_result_of_{dataset}_training_by_PPF')
        result_FPIR = pd.read_csv(f'Result/Result_after_merge/All_result_of_{dataset}_training_by_FPIR')

        #pre
        FUDS_acc = [result_FUDS[t::10]['acc'].mean() for t in range(10)]
        FUDS_ddp = [result_FUDS[t::10]['disparity'].mean() for t in range(10)]
        FAWOS_acc = [result_FAWOS[t::10]['acc'].mean() for t in range(10)]
        FAWOS_ddp = [result_FAWOS[t::10]['disparity'].mean() for t in range(10)]
        DIR_acc= [result_DIR[t::10]['acc'].mean() for t in range(10)]
        DIR_ddp = [result_DIR[t::10]['disparity'].mean() for t in range(10)]


        #in
        FCSC_acc = [result_FCSC[t::10]['acc'].mean() for t in range(10)]
        FCSC_ddp = [result_FCSC[t::10]['disparity'].mean() for t in range(10)]
        KDE_acc = [result_KDE[t::10]['acc'].mean() for t in range(10)]
        KDE_ddp = [result_KDE[t::10]['disparity'].mean() for t in range(10)]
        ADV_acc = [result_ADV[t::10]['acc'].mean() for t in range(10)]
        ADV_ddp = [np.abs(result_ADV[t::10]['disparity']).mean() for t in range(10)]

        #post
        FPIR_acc = [result_FPIR[t::10]['acc'].mean() for t in range(10)]
        FPIR_ddp = [result_FPIR[t::10]['disparity'].mean() for t in range(10)]
        PPOT_acc = [result_PPOT[t::10]['acc'].mean() for t in range(10)]
        PPOT_ddp = [result_PPOT[t::10]['disparity'].mean() for t in range(10)]
        PPF_acc = [result_PPF[t::10]['acc'].mean() for t in range(10)]
        PPF_ddp = [result_PPF[t::10]['disparity'].mean() for t in range(10)]


        plt.figure(figsize=(19, 6), dpi=100)
        #
        plt.subplot(1,3,1)
        plt.scatter(FUDS_acc,FUDS_ddp,marker = '*', s =400, label=r'FUDS')
        plt.scatter(DIR_acc,DIR_ddp, marker = 'x',s =400, label=r'DIR')
        plt.scatter(FAWOS_acc,  FAWOS_ddp,marker = '+', s =400, label=r'FAWOS')


        plt.xlabel('Accuracy',fontsize = 25)
        plt.ylabel('Demographic Disparity',fontsize = 25)
        # plt.text(8,0.04,r'$\times 100$', fontsize = 20)
        plt.legend( fontsize = 22)
        plt.title('Pre-processing',fontsize = 25)
        # plt.ylim(0.01,0.21)
        # plt.xlim(0.823,0.855)
        plt.xticks( xticks_pre,fontsize = 23)
        plt.yticks( yticks,fontsize = 23)
        plt.subplot(1,3,2)


        plt.scatter(FCSC_acc,FCSC_ddp, marker = '*',s =400, label=r'FCSC')
        plt.scatter(KDE_acc,KDE_ddp,marker='x', s =400, label=r'KDE')
        plt.scatter(ADV_acc,ADV_ddp,marker='+', s =400, label=r'ADV')
        plt.xlabel('Accuracy',fontsize = 25)
        # plt.ylabel('DDP',fontsize = 26)
        # plt.text(8,0.04,r'$\times 100$', fontsize = 20)

        plt.title('In-processing',fontsize = 25)
        plt.legend( fontsize = 22)
        # plt.ylim(0.01,0.21)
        # plt.xlim(0.825,0.855)
        plt.xticks( xticks_in,fontsize = 23)
        plt.yticks( yticks,fontsize = 23)
        # plt.yticks( fontsize = 25)
        plt.subplot(1,3,3)



        plt.scatter(FPIR_acc,FPIR_ddp,  marker = '*',s =400, label=r'FPIR')
        plt.scatter(PPF_acc,PPF_ddp,marker = 'x', s =400, label=r'PPF')
        plt.scatter(PPOT_acc,PPOT_ddp,marker = '+', s =400, label=r'PPOT')
        plt.xlabel('Accuracy',fontsize = 25)
        # plt.ylabel('DDP',fontsize = 26)
        # plt.text(8,0.04,r'$\times 100$', fontsize = 20)
        plt.legend(loc='upper left', fontsize = 22)
        # plt.ylim(0.01,0.21)
        # plt.xlim(0.825,0.855)
        plt.xticks( xticks_post,fontsize = 23)
        plt.yticks( yticks,fontsize = 23)
        plt.title('Post-processing',fontsize = 25)

        #plt.yticks( [0,0.05,0.1,0.15,0.2],fontsize = 26)
        plt.tight_layout()
        plt.savefig(f'{dataset}')

