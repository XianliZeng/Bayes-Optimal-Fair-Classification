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

def FCSC(dataset,dataset_name,net, optimizer,lr_schedule, delta, device, n_epochs=200, batch_size=2048, seed=0):

    # Retrieve train/test splitted pytorch tensors for index=split
    training_tensors, testing_tensors = dataset.get_dataset_in_tensor()
    X_train, Y_train, Z_train, XZ_train = training_tensors
    X_test, Y_test, Z_test, XZ_test = testing_tensors


    XZ_train11 = XZ_train[(Z_train == 1) & (Y_train == 1)]
    XZ_train10 = XZ_train[(Z_train == 1) & (Y_train == 0)]
    XZ_train01 = XZ_train[(Z_train == 0) & (Y_train == 1)]
    XZ_train00 = XZ_train[(Z_train == 0) & (Y_train == 0)]



    n11_ori = len(XZ_train11)
    n10_ori = len(XZ_train10)
    n01_ori = len(XZ_train01)
    n00_ori = len(XZ_train00)
    n = len(X_train)


    p11, p10, p01, p00 = n11_ori / n, n10_ori / n, n01_ori / n, n00_ori / n
    tmin, tmax = cal_t_bound(p11, p10, p01, p00 )

    tmid = 0


    custom_dataset = CustomDataset(XZ_train, Y_train, Z_train)
    if batch_size == 'full':
        batch_size_ = XZ_train.shape[0]
    elif isinstance(batch_size, int):
        batch_size_ = batch_size
    data_loader = DataLoader(custom_dataset, batch_size=batch_size_, shuffle=True)
    loss_function = nn.BCELoss()

    #####Pre-train the model######
    with tqdm(range(n_epochs//4)) as epochs:
        epochs.set_description(f"Pre-train the model:  dataset: {dataset_name}, seed: {seed}, level:{delta}")

        for epoch in epochs:
            net.train()
            for i, (xz_batch, y_batch, z_batch) in enumerate(data_loader):
                xz_batch, y_batch, z_batch = xz_batch.to(device), y_batch.to(device), z_batch.to(device)
                Yhat = net(xz_batch)

                cost = loss_function(Yhat.squeeze(), y_batch)

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                epochs.set_postfix(loss=cost.item())
            if dataset_name == 'AdultCensus':
                lr_schedule.step()

    weightspg = [1/4,1/4,1/4,1/4]

    ########Update the threshold parameter#######
    for T in range(15):
        loss_function = nn.BCELoss(reduction = 'none')

        with tqdm(range(n_epochs//20)) as epochs:
            epochs.set_description(f"training with dataset: {dataset_name}, seed: {seed}, level: {delta}, T: {T}")

            for epoch in epochs:
                net.train()
                for i, (xz_batch, y_batch, z_batch) in enumerate(data_loader):
                    xz_batch, y_batch, z_batch = xz_batch.to(device), y_batch.to(device), z_batch.to(device)
                    Yhat = net(xz_batch)
                    weights = torch.zeros_like(y_batch)
                    weights[(z_batch==1) &(y_batch==1)] = weightspg[0]
                    weights[(z_batch==1) &(y_batch==0)] = weightspg[1]
                    weights[(z_batch==0) &(y_batch==1)] = weightspg[2]
                    weights[(z_batch==0) &(y_batch==0)] = weightspg[3]

                    cost_initial = loss_function(Yhat.squeeze(), y_batch)
                    cost = torch.mean(cost_initial * weights)
                    optimizer.zero_grad()
                    cost.backward()
                    optimizer.step()

                    epochs.set_postfix(loss=cost.item())
                if dataset_name == 'AdultCensus':
                    lr_schedule.step()

        with torch.no_grad():
            Yhat_train = net(XZ_train).detach().squeeze().numpy()>0.5
            disparity = Yhat_train[Z_train == 1].mean() - Yhat_train[Z_train == 0].mean()

            if disparity > delta:
                tmin =  tmid
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

        T1,T0 = cal_threshold(tmid,p11,p10,p01,p00)
        weightspg = [(1-T1)/2, (T1)/2, (1-T0)/2, T0/2]

####Evaluate performance##########


    eta_test = net(XZ_test).detach().cpu().numpy().squeeze()


    Y_test_np = Y_test.clone().detach().numpy()
    Z_test_np = Z_test.clone().detach().numpy()


    acc = cal_acc(eta_test,Y_test_np,Z_test_np,0.5,0.5)
    disparity = cal_disparity(eta_test,Z_test_np,0.5,0.5)

    data = [seed,dataset_name,delta,acc, np.abs(disparity)]
    columns = ['seed','dataset','level','acc', 'disparity']


    df_test = pd.DataFrame([data], columns=columns)

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
    return n_epochs,lr,batch_size






def training_FCSC(dataset_name,delta,seed):
    print(f'we are running dataset_name: {dataset_name} with seed: {seed}')
    device = torch.device('cpu')

    # Set a seed for random number generation
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Import dataset
    dataset = FairnessDataset(dataset=dataset_name, seed = seed, device=device)
    dataset.normalize()
    input_dim = dataset.XZ_train.shape[1]
    n_epochs, lr, batch_size= get_training_parameters(dataset_name)
    # Create a classifier model




    net = Classifier(n_inputs=input_dim)
    net = net.to(device)
    lr_decay = 0.98
    # Set an optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # Fair classifier training
    lr_schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    Result = FCSC(dataset=dataset,dataset_name=dataset_name,
                 net=net,
                 optimizer=optimizer,lr_schedule=lr_schedule,delta = delta,
                 device=device, n_epochs=n_epochs, batch_size=batch_size, seed=seed)
    print(Result)
    Result.to_csv(f'Result/FCSC/result_of_{dataset_name}_with_seed_{seed}_para_{int(delta*1000)}')










