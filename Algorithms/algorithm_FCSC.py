import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from DataLoader.dataloader import CustomDataset
from utils import cal_disparity,cal_acc,cal_t_bound, cal_threshold
import time

from DataLoader.dataloader import FairnessDataset
import torch.optim as optim
from models import Classifier

from sklearn.ensemble import HistGradientBoostingClassifier

class Classifier(nn.Module):
    def __init__(self, n_inputs):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
                        nn.Linear(n_inputs, 32),
                        nn.ReLU(),
                        nn.Linear(32, 32),
                        nn.ReLU(),
                        nn.Linear(32, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                        nn.Sigmoid()
                    )

    def forward(self, x):
        predict = self.model(x)
        return predict

import resource

def thread_cpu_time():
    r = resource.getrusage(resource.RUSAGE_THREAD)
    return r.ru_utime + r.ru_stime

from sklearn.ensemble import HistGradientBoostingClassifier

def FCSC(dataset, dataset_name, blind, net, optimizer, lr_schedule, delta, device,
         n_epochs=200, batch_size=2048, seed=0, model="mlp"):
    start_time = thread_cpu_time()

    training_tensors, testing_tensors = dataset.get_dataset_in_tensor()
    X_train, Y_train, Z_train, XZ_train = training_tensors
    X_test, Y_test, Z_test, XZ_test = testing_tensors

    if blind:
        X_fit = X_train.cpu().numpy()
    else:
        X_fit = XZ_train.cpu().numpy()

    y_fit = Y_train.squeeze(-1).cpu().numpy()
    z_fit = Z_train.squeeze(-1).cpu().numpy()

    if model == "hgb":
        # Partition sizes
        n = len(X_fit)
        idx11 = (z_fit == 1) & (y_fit == 1)
        idx10 = (z_fit == 1) & (y_fit == 0)
        idx01 = (z_fit == 0) & (y_fit == 1)
        idx00 = (z_fit == 0) & (y_fit == 0)

        p11 = np.sum(idx11) / n
        p10 = np.sum(idx10) / n
        p01 = np.sum(idx01) / n
        p00 = np.sum(idx00) / n

        tmin, tmax = cal_t_bound(p11, p10, p01, p00)
        tmid = 0
        weightspg = [1/4, 1/4, 1/4, 1/4]

        for T in range(15):
            # Assign sample weights according to current group weights
            weights = np.zeros_like(y_fit, dtype=float)
            weights[idx11] = weightspg[0]
            weights[idx10] = weightspg[1]
            weights[idx01] = weightspg[2]
            weights[idx00] = weightspg[3]

            clf = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=300,
                max_depth=4,
                l2_regularization=1.0,
                random_state=seed
            )
            clf.fit(X_fit, y_fit, sample_weight=weights)

            if blind:
                X_eval = X_test.cpu().numpy()
            else:
                X_eval = XZ_test.cpu().numpy()

            eta_test = clf.predict_proba(X_eval)[:, 1]

            # Evaluate disparity and update thresholds
            disparity = eta_test[Z_test == 1].mean() - eta_test[Z_test == 0].mean()
            if disparity > delta:
                tmin = tmid
            elif disparity < -delta:
                tmax = tmid
            elif disparity > 0:
                if tmid > 0:
                    tmax = tmid
                else:
                    tmin = tmid
            else:
                if tmid > 0:
                    tmax = tmid
                else:
                    tmin = tmid

            tmid = (tmax + tmin) / 2
            T1, T0 = cal_threshold(tmid, p11, p10, p01, p00)
            weightspg = [(1-T1)/2, T1/2, (1-T0)/2, T0/2]

        # Final prediction
        eta_test = clf.predict_proba(X_eval)[:, 1]

    else:
        # Fallback to existing MLP FCSC implementation (unchanged)
        custom_dataset = CustomDataset(XZ_train if not blind else X_train, Y_train, Z_train)
        batch_size_ = batch_size if isinstance(batch_size, int) else len(custom_dataset)
        data_loader = DataLoader(custom_dataset, batch_size=batch_size_, shuffle=True)
        loss_function = nn.BCELoss()

        # Pretrain
        for epoch in tqdm(range(n_epochs // 4), desc=f"Pretrain: {dataset_name}, seed={seed}"):
            net.train()
            for xb, yb, zb in data_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                yhat = net(xb).squeeze()
                loss = loss_function(yhat, yb)
                loss.backward()
                optimizer.step()
            if dataset_name == 'AdultCensus':
                lr_schedule.step()

        # Dynamic cost-sensitive training
        tmid = 0
        weightspg = [1/4, 1/4, 1/4, 1/4]
        for T in range(15):
            net.train()
            loss_function = nn.BCELoss(reduction='none')
            for epoch in tqdm(range(n_epochs // 20), desc=f"T={T}"):
                for xb, yb, zb in data_loader:
                    xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)
                    yhat = net(xb).squeeze()
                    weights = torch.zeros_like(yb)
                    weights[(zb==1)&(yb==1)] = weightspg[0]
                    weights[(zb==1)&(yb==0)] = weightspg[1]
                    weights[(zb==0)&(yb==1)] = weightspg[2]
                    weights[(zb==0)&(yb==0)] = weightspg[3]
                    loss = torch.mean(loss_function(yhat, yb) * weights)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if dataset_name == 'AdultCensus':
                    lr_schedule.step()

            # Evaluate disparity
            with torch.no_grad():
                X_eval = XZ_test if not blind else X_test
                eta_test = net(X_eval).squeeze().cpu().numpy()
                disparity = eta_test[Z_test == 1].mean() - eta_test[Z_test == 0].mean()
                if disparity > delta:
                    tmin = tmid
                elif disparity < -delta:
                    tmax = tmid
                elif disparity > 0:
                    if tmid > 0: tmax = tmid
                    else: tmin = tmid
                else:
                    if tmid > 0: tmax = tmid
                    else: tmin = tmid
                tmid = (tmax + tmin) / 2
                T1, T0 = cal_threshold(tmid, p11, p10, p01, p00)
                weightspg = [(1-T1)/2, T1/2, (1-T0)/2, T0/2]

    # Final evaluation
    Y_test_np = Y_test.cpu().numpy()
    Z_test_np = Z_test.cpu().numpy()
    acc = cal_acc(eta_test, Y_test_np, Z_test_np, 0.5, 0.5)
    disparity = cal_disparity(eta_test, Z_test_np, 0.5, 0.5)
    df_test = pd.DataFrame([[seed, dataset_name, delta, acc, abs(disparity), thread_cpu_time() - start_time]],
                           columns=['seed', 'dataset', 'level', 'acc', 'disparity', 'time'])
    return df_test



def get_training_parameters(dataset_name):
    if dataset_name == 'AdultCensus':
        n_epochs = 200
        lr = 1e-1
        batch_size = 512
        ##### predetermine disparity level #####
        #
        # delta_set_dp = np.arange(0, 50, 1) / 200
        # delta_set_eo = np.arange(0, 50, 1) / 250

    if dataset_name == 'COMPAS':
        n_epochs = 500
        lr = 5e-4
        batch_size = 2048
        ##### predetermine disparity level #####
        #
        # delta_set_dp = np.arange(0, 50, 1) / 150
        # delta_set_eo = np.arange(0, 50, 1) / 140

    if dataset_name == 'Lawschool':
        n_epochs = 200
        lr = 2e-4
        batch_size = 2048
        ##### predetermine disparity level #####

        # delta_set_dp = np.arange(0, 50, 1) / 500
        # delta_set_eo = np.arange(0, 50, 1) / 340
        
    if dataset_name == 'ACSIncome':
            n_epochs = 20
            lr = 1e-3
            batch_size = 128
    return n_epochs,lr,batch_size






def training_FCSC(dataset_name,delta,blind,seed,model):
    print(f'we are running dataset_name: {dataset_name} with seed: {seed}')
    device = torch.device('cpu')

    # Set a seed for random number generation
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Import dataset
    dataset = FairnessDataset(dataset=dataset_name, seed = seed, device=device)
    dataset.normalize()
    if not blind:
        input_dim = dataset.XZ_train.shape[1]
    elif blind:
        input_dim = dataset.X_train.shape[1]
    n_epochs, lr, batch_size= get_training_parameters(dataset_name)
    # Create a classifier model




    net = Classifier(n_inputs=input_dim)
    net = net.to(device)
    lr_decay = 0.98
    # Set an optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # Fair classifier training
    lr_schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    Result = FCSC(dataset=dataset,dataset_name=dataset_name,blind=blind,
                 net=net,
                 optimizer=optimizer,lr_schedule=lr_schedule,delta = delta,
                 device=device, n_epochs=n_epochs, batch_size=batch_size, seed=seed, model=model)
    print(Result)
    if not blind:
        Result.to_csv(f'Result/FCSC/{model.upper()}/result_of_{dataset_name}_with_seed_{seed}_para_{int(delta*1000)}')
    elif blind:
        Result.to_csv(f'Result/FCSC/{model.upper()}/blind/result_of_{dataset_name}_with_seed_{seed}_para_{int(delta*1000)}')










