import random
import IPython
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from DataLoader.dataloader import CustomDataset
from utils import cal_disparity,cal_acc,cal_t_bound, number_of_sample
import time

from DataLoader.dataloader import FairnessDataset
import torch.optim as optim
from models import Classifier

class Classifier(nn.Module):
    def __init__(self, n_inputs):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
                        nn.Linear(n_inputs, 500),
                        nn.ReLU(),
                        nn.Linear(500, 200),
                        nn.ReLU(),
                        nn.Linear(200, 100),
                        nn.ReLU(),
                        nn.Linear(100, 1),
                        nn.Sigmoid()
                    )

    def forward(self, x):
        predict = self.model(x)
        return predict

import resource

def thread_cpu_time():
    r = resource.getrusage(resource.RUSAGE_THREAD)
    return r.ru_utime + r.ru_stime

def get_level(dataset_name,fairness):
    if dataset_name == 'AdultCensus':
        if fairness == 'DP':
            level_list = np.arange(50)/250
        if fairness == 'EO':
            level_list = np.arange(50) / 1250
        if fairness == 'PE':
            level_list = np.arange(10) / 250
        if fairness == 'OAE':
            level_list = np.arange(10) / 2000

    if dataset_name == 'COMPAS':
        if fairness == 'DP':
            level_list = np.arange(10)/50
        if fairness == 'EO':
            level_list = np.arange(10) /50
        if fairness == 'PE':
            level_list = np.arange(10) /50
        if fairness == 'OAE':
            level_list = np.arange(10) / 500
    if dataset_name == 'Lawschool':
        if fairness == 'DP':
            level_list = np.arange(10)/250
        if fairness == 'EO':
            level_list = np.arange(10) /125
        if fairness == 'PE':
            level_list = np.arange(10) /2000
        if fairness == 'OAE':
            level_list = np.arange(10) / 200
    if dataset_name == 'ACSIncome':
        if fairness == 'DP':
              level_list = 0.025 * np.arange(10)
    return level_list

def FUDS(dataset,dataset_name,blind,
         net, optimizer,lr_schedule, delta, device, n_epochs=200, batch_size=2048, seed=0):

    training_tensors, testing_tensors = dataset.get_dataset_in_tensor()
    X_train, Y_train, Z_train, XZ_train = training_tensors
    X_test, Y_test, Z_test, XZ_test = testing_tensors


    Y_train_np = Y_train.clone().detach().numpy()
    Z_train_np = Z_train.clone().detach().numpy()

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

    n11_now, n10_now, n01_now, n00_now = n11_ori, n10_ori, n01_ori, n00_ori
    XZ_train11_now = XZ_train11.clone()
    XZ_train10_now = XZ_train10.clone()
    XZ_train01_now = XZ_train01.clone()
    XZ_train00_now = XZ_train00.clone()

    if not blind:
        custom_dataset = CustomDataset(XZ_train, Y_train, Z_train)
        if batch_size == 'full':
            batch_size_ = XZ_train.shape[0]
        elif isinstance(batch_size, int):
            batch_size_ = batch_size
    elif blind:
        custom_dataset = CustomDataset(X_train, Y_train, Z_train)
        if batch_size == 'full':
            batch_size_ = X_train.shape[0]
        elif isinstance(batch_size, int):
            batch_size_ = batch_size
    data_loader = DataLoader(custom_dataset, batch_size=batch_size_, shuffle=True)
    loss_function = nn.BCELoss()

#####Pre-train the model######
    start_time = thread_cpu_time()
    with tqdm(range(n_epochs//4)) as epochs:
        epochs.set_description(f"Pre-train the model:  dataset: {dataset_name}, seed: {seed}, level:{delta}")

        for epoch in epochs:
            net.train()
            for i, (covariate_batch, y_batch, z_batch) in enumerate(data_loader):
                
                covariate_batch, y_batch, z_batch = covariate_batch.to(device), y_batch.to(device), z_batch.to(device)
                
                Yhat = net(covariate_batch)

                cost = loss_function(Yhat.squeeze(), y_batch)

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                epochs.set_postfix(loss=cost.item())
            if dataset_name == 'AdultCensus':
                lr_schedule.step()


########Update the threshold parameter#######
    for T in range(15):
        loss_function = nn.BCELoss()

        with tqdm(range(n_epochs//20)) as epochs:
            epochs.set_description(f"training with dataset: {dataset_name}, seed: {seed}, level:{delta}, T: {T}")

            for epoch in epochs:
                net.train()
                for i, (covariate_batch, y_batch, z_batch) in enumerate(data_loader):
                    covariate_batch, y_batch, z_batch = covariate_batch.to(device), y_batch.to(device), z_batch.to(device)
                    y_batch = y_batch.unsqueeze(1)
                    z_batch = z_batch.unsqueeze(1)

                    Yhat = net(covariate_batch)

                    cost = loss_function(Yhat, y_batch)

                    optimizer.zero_grad()
                    cost.backward()
                    optimizer.step()

                    epochs.set_postfix(loss=cost.item())

                if dataset_name == 'AdultCensus':
                    lr_schedule.step()

    ########choose the model with best performance on validation set###########
        with torch.no_grad():
            if not blind:
                Yhat_train = net(XZ_train).detach().squeeze().numpy()>0.5
            elif blind:
                Yhat_train = net(X_train).detach().squeeze().numpy()>0.5
            disparity = Yhat_train[Z_train == 1].mean() - Yhat_train[Z_train == 0].mean()

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
        n11, n10, n01, n00 = number_of_sample(n11_ori, n10_ori, n01_ori, n00_ori,  tmid, n, blind)

        if n11 > n11_now:
            if n11 > (n11_now + n11_ori):
                index11 = np.random.choice(len(XZ_train11), n11 - n11_now, replace=True)
                XZ_syn11 = XZ_train11[index11, :]
                XZ_train11_now = torch.concat([XZ_train11_now, XZ_syn11])
            else:
                index11 = np.random.choice(len(XZ_train11), n11 - n11_now, replace=False)
                XZ_syn11 = XZ_train11[index11, :]
                XZ_train11_now = torch.concat([XZ_train11_now, XZ_syn11])
        else:
            index11 = np.random.choice(len(XZ_train11_now), n11, replace=False)
            XZ_train11_now = XZ_train11_now[index11, :]

        if n10 > n10_now:
            if n10 > (n10_now + n10_ori):
                index10 = np.random.choice(len(XZ_train10), n10 - n10_now, replace=True)
                XZ_syn10 = XZ_train10[index10, :]
                XZ_train10_now = torch.concat([XZ_train10_now, XZ_syn10])
            else:
                index10 = np.random.choice(len(XZ_train10), n10 - n10_now, replace=False)
                XZ_syn10 = XZ_train10[index10, :]
                XZ_train10_now = torch.concat([XZ_train10_now, XZ_syn10])
        else:
            index10 = np.random.choice(len(XZ_train10_now), n10, replace=False)
            XZ_train10_now = XZ_train10_now[index10, :]

        if n01 > n01_now:
            if n01 > (n01_now + n01_ori):
                index01 = np.random.choice(len(XZ_train01), n01 - n01_now, replace=True)
                XZ_syn01 = XZ_train01[index01, :]
                XZ_train01_now = torch.concat([XZ_train01_now, XZ_syn01])
            else:
                index01 = np.random.choice(len(XZ_train01), n01 - n01_now, replace=False)
                XZ_syn01 = XZ_train01[index01, :]
                XZ_train01_now = torch.concat([XZ_train01_now, XZ_syn01])
        else:
            index01 = np.random.choice(len(XZ_train01_now), n01, replace=False)
            XZ_train01_now = XZ_train01_now[index01, :]

        if n00 > n00_now:
            if n00 > (n00_now + n00_ori):
                index00 = np.random.choice(len(XZ_train00), n00 - n00_now, replace=True)
                XZ_syn00 = XZ_train00[index00, :]
                XZ_train00_now = torch.concat([XZ_train00_now, XZ_syn00])
            else:
                index00 = np.random.choice(len(XZ_train00), n00 - n00_now, replace=False)
                XZ_syn00 = XZ_train00[index00, :]
                XZ_train00_now = torch.concat([XZ_train00_now, XZ_syn00])
        else:
            index00 = np.random.choice(len(XZ_train00_now), n00, replace=False)
            XZ_train00_now = XZ_train00_now[index00, :]

        Y_syn11 = torch.ones(len(XZ_train11_now))
        Y_syn10 = torch.zeros(len(XZ_train10_now))
        Y_syn01 = torch.ones(len(XZ_train01_now))
        Y_syn00 = torch.zeros(len(XZ_train00_now))

        Z_syn11 = torch.ones(len(XZ_train11_now))
        Z_syn10 = torch.ones(len(XZ_train10_now))
        Z_syn01 = torch.zeros(len(XZ_train01_now))
        Z_syn00 = torch.zeros(len(XZ_train00_now))

        XZ_syn = torch.concat([XZ_train11_now, XZ_train10_now, XZ_train01_now, XZ_train00_now])
        Y_syn = torch.concat([Y_syn11, Y_syn10, Y_syn01, Y_syn00])
        Z_syn = torch.concat([Z_syn11, Z_syn10, Z_syn01, Z_syn00])

        if not blind:
            custom_dataset = CustomDataset(XZ_syn, Y_syn, Z_syn)
            if batch_size == 'full':
                batch_size_ = XZ_train.shape[0]
            elif isinstance(batch_size, int):
                batch_size_ = batch_size
            data_loader = DataLoader(custom_dataset, batch_size=batch_size_, shuffle=True)
        elif blind:
            X_syn = XZ_syn[:, :-1]  #remove last column which is protected attribute
            custom_dataset = CustomDataset(X_syn, Y_syn, Z_syn)
            if batch_size == 'full':
                batch_size_ = X_train.shape[0]
            elif isinstance(batch_size, int):
                batch_size_ = batch_size
            data_loader = DataLoader(custom_dataset, batch_size=batch_size_, shuffle=True)

        n11_now, n10_now, n01_now, n00_now = n11, n10, n01, n00

####Evaluate performance##########
    if not blind:
        eta_test = net(XZ_test).detach().cpu().numpy().squeeze()
    elif blind:
        eta_test = net(X_test).detach().cpu().numpy().squeeze()

    Y_test_np = Y_test.clone().detach().numpy()
    Z_test_np = Z_test.clone().detach().numpy()

    acc = cal_acc(eta_test,Y_test_np,Z_test_np,0.5,0.5)
    disparity = cal_disparity(eta_test,Z_test_np,0.5,0.5)

    data = [seed,dataset_name,delta,acc, np.abs(disparity), thread_cpu_time() - start_time]
    columns = ['seed','dataset','level','acc', 'disparity', 'time']
    df_test = pd.DataFrame([data], columns=columns)
    return df_test





def get_training_parameters(dataset_name):
    if dataset_name == 'AdultCensus':
        n_epochs = 200
        lr = 1e-1
        batch_size = 512

    if dataset_name == 'COMPAS':
        n_epochs = 500
        lr = 5e-4
        batch_size = 2048

    if dataset_name == 'Lawschool':
        n_epochs = 200
        lr = 2e-4
        batch_size = 2048
        
    if dataset_name == 'ACSIncome':
        n_epochs = 20
        lr = 1e-3
        batch_size = 128
        
    return n_epochs,lr,batch_size


def training_FUDS(dataset_name, delta, blind, seed):
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

    # Set an optimizer and decay rate
    lr_decay = 0.98
    optimizer = optim.Adam(net.parameters(), lr=lr)
    lr_schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    # Fair classifier training
    Result = FUDS(dataset=dataset, dataset_name=dataset_name, blind=blind,
                     net=net,
                     optimizer=optimizer,lr_schedule=lr_schedule,delta = delta,
                     device=device, n_epochs=n_epochs, batch_size=batch_size, seed=seed)

    print(Result)
    if not blind:
        Result.to_csv(f'Result/FUDS/NNo/result_of_{dataset_name}_with_seed_{seed}_para_{int(delta*1000)}')
    elif blind:
        Result.to_csv(f'Result/FUDS/NNo/result_of_{dataset_name}_with_seed_{seed}_para_{int(delta*1000)}_blind')











