import random
import IPython
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from DataLoader.dataloader_FAWOS import CustomDataset
from utils import cal_disparity,cal_acc, sample_points
import time

from DataLoader.dataloader_FAWOS import FairnessDataset
import torch.optim as optim
from models import Classifier

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

def FAWOS(dataset,dataset_name,blind, net, optimizer,lr_schedule, alpha, device, n_epochs=200, batch_size=2048, seed=0):

    # Retrieve train/test splitted pytorch tensors for index=split
    df_train,df_test = dataset.get_dataset_in_dataframe()

    Z_train = df_train['Sen1']
    Y_train = df_train['Label']


    df_train11 = df_train[(Z_train == 1) & (Y_train == 1)]
    df_train10 = df_train[(Z_train == 1) & (Y_train == 0)]
    df_train01 = df_train[(Z_train == 0) & (Y_train == 1)]
    df_train00 = df_train[(Z_train == 0) & (Y_train == 0)]


    n11_ori = len(df_train11)
    n10_ori = len(df_train10)
    n01_ori = len(df_train01)
    n00_ori = len(df_train00)

    if dataset_name == 'COMPAS':
        n00 = round(alpha * (n10_ori * n01_ori / n11_ori - n00_ori))
        df_generated = sample_points(df_train00, df_train, n00)
    else:
        n01 = round(alpha * (n11_ori * n00_ori / n10_ori - n01_ori))
        df_generated = sample_points(df_train01, df_train, n01)
    df_syn = pd.concat([df_train,df_generated])
    Y_syn = df_syn['Label'].values
    Z_syn = df_syn['Sen1'].values
    XZ_syn = df_syn.drop(['weight', 'neighbour', 'Label', 'Sen1'], axis=1).values
    X_syn = df_syn.drop(['weight', 'neighbour', 'Label', 'Sen1', 'Sensitive'], axis=1).values
    
    Y_syn = torch.FloatTensor(Y_syn).to(device)
    Z_syn = torch.FloatTensor(Z_syn).to(device)
    XZ_syn = torch.FloatTensor(XZ_syn).to(device)
    X_syn = torch.FloatTensor(X_syn).to(device)


    Y_test = df_test['Label'].values
    Z_test = df_test['Sen1'].values
    XZ_test = df_test.drop(['Label', 'Sen1'], axis=1).values
    X_test = df_test.drop(['Label', 'Sen1', 'Sensitive'], axis=1).values
    X_test = torch.FloatTensor(X_test).to(device)
    XZ_test = torch.FloatTensor(XZ_test).to(device)
    




    if not blind:
        custom_dataset = CustomDataset(XZ_syn, Y_syn, Z_syn)
        if batch_size == 'full':
            batch_size_ = XZ_syn.shape[0]
        elif isinstance(batch_size, int):
            batch_size_ = batch_size
        data_loader = DataLoader(custom_dataset, batch_size=batch_size_, shuffle=True)
    elif blind: 
        custom_dataset = CustomDataset(X_syn, Y_syn, Z_syn)
        if batch_size == 'full':
            batch_size_ = X_syn.shape[0]
        elif isinstance(batch_size, int):
            batch_size_ = batch_size
        data_loader = DataLoader(custom_dataset, batch_size=batch_size_, shuffle=True)
    
    loss_function = nn.BCELoss()

    start_time = thread_cpu_time()

    with tqdm(range(n_epochs)) as epochs:
        epochs.set_description(f"Training with dataset: {dataset_name}, seed: {seed}, alpha: {alpha}")

        for epoch in epochs:
            net.train()
            for i, (covariate_batch, y_batch, z_batch) in enumerate(data_loader):
                covariate_batch, y_batch, z_batch = covariate_batch.to(device), y_batch.to(device), z_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(1)
                z_batch = z_batch.to(device).unsqueeze(1)
                Yhat = net(covariate_batch)

                # prediction loss
                cost = loss_function(Yhat, y_batch)

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                # Print the cost per 10 batches
                epochs.set_postfix(loss=cost.item())
            if dataset_name == 'AdultCensus':
                lr_schedule.step()

####Performance Evaluation
    if not blind:
        eta_test = net(XZ_test).detach().cpu().numpy().squeeze()
    elif blind:
        eta_test = net(X_test).detach().cpu().numpy().squeeze()

    df_test = pd.DataFrame()


    acc = cal_acc(eta_test,Y_test,Z_test,0.5,0.5)
    disparity = cal_disparity(eta_test,Z_test,0.5,0.5)

    data = [seed,dataset_name,alpha,acc, np.abs(disparity), thread_cpu_time()-start_time]
    columns = ['seed','dataset','alpha','acc', 'disparity', 'time']


    temp_dp = pd.DataFrame([data], columns=columns)
    df_test= pd.concat([df_test,temp_dp])

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






def training_FAWOS(dataset_name,alpha,blind,seed):
    print(f'we are running dataset_name: {dataset_name} with seed: {seed}')
    device = torch.device('cpu')


    # Set a seed for random number generation
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = FairnessDataset(dataset=dataset_name, seed=seed, device=device)
    dataset.normalize()
    if not blind:
        input_dim = dataset.XZ_train.shape[1]
    elif blind:
        input_dim = dataset.X_train.shape[1]
    n_epochs, lr, batch_size = get_training_parameters(dataset_name)
    # Import dataset
        # Create a classifier model
    net = Classifier(n_inputs=input_dim)
    net = net.to(device)

# Set an optimizer
    lr_decay = 0.98
    optimizer = optim.Adam(net.parameters(), lr=lr)
    lr_schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
# Fair classifier training

    Result = FAWOS(dataset=dataset,dataset_name=dataset_name,blind=blind,
                     net=net,
                     optimizer=optimizer,lr_schedule = lr_schedule,alpha = alpha,
                     device=device, n_epochs=n_epochs, batch_size=batch_size, seed=seed)
    print(Result)

    if not blind:
        Result.to_csv(f'Result/FAWOS/result_of_{dataset_name}_with_seed_{seed}_para_{int(alpha*1000)}')
    elif blind:
        Result.to_csv(f'Result/FAWOS/result_of_{dataset_name}_with_seed_{seed}_para_{int(alpha*1000)}_blind')









