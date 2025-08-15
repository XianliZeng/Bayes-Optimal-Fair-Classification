import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from DataLoader.dataloader import CustomDataset
from utils import threshold_flipping, cal_acc_PPF, cal_disparity_PPF, postprocessing_flipping
import time
from sklearn.model_selection import train_test_split

from DataLoader.dataloader import FairnessDataset
import torch.optim as optim
#from models import Classifier, multi_Classifier

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

def get_level(dataset_name):
    if dataset_name == 'AdultCensus':
        level_list = np.arange(10)/50
        level_list[0]=0.001
    if dataset_name == 'COMPAS':
        level_list  = 0.3 * np.arange(10) / 10

    if dataset_name == 'Lawschool':
        level_list = np.arange(10) / 125
        level_list[0]=0.001
    
    if dataset_name == 'ACSIncome':
        level_list = 0.025 * np.arange(10)
        level_list[0]=0.001

    return level_list

def PPF(dataset,dataset_name,blind, net, optimizer, lr_schedule,device, n_epochs=200, batch_size=2048, seed=0):

    training_tensors, testing_tensors = dataset.get_dataset_in_tensor()
    X_train, Y_train, Z_train, XZ_train = training_tensors
    X_test, Y_test, Z_test, XZ_test = testing_tensors

    #
    Y_train_np = Y_train.detach().cpu().numpy()
    Z_train_np = Z_train.detach().cpu().numpy()

    Y_test_np = Y_test.clone().detach().numpy()
    Z_test_np = Z_test.clone().detach().numpy()

    #### if DDP<0, let Z=1-Z####
    dp0 = (Y_train_np[Z_train_np==1]).mean() - (Y_train_np[Z_train_np==0]).mean()
    if dp0<0:
        Z_train = 1 - Z_train
        Z_train_np = 1 - Z_train_np
        Z_test_np = 1 - Z_test_np

    if not blind:
        custom_dataset = CustomDataset(XZ_train, Y_train, Z_train)
        if batch_size == 'full':
            batch_size_ = XZ_train.shape[0]
        elif isinstance(batch_size, int):
            batch_size_ = batch_size
        data_loader = DataLoader(custom_dataset, batch_size=batch_size_, shuffle=True)
        loss_function = nn.BCELoss()
    elif blind:
        X_train, X_postproc, Y_train, Y_postproc, Z_train, Z_postproc, XZ_train, XZ_postproc = train_test_split(
            X_train, Y_train, Z_train, XZ_train, test_size=0.30, random_state=seed) # Further split the training set into training and post-processing
        # Train AY jointly, for blind setting
        Z_Y_train = torch.tensor(2 * Z_train.clone().detach() + Y_train.clone().detach(), dtype=torch.long)
        custom_dataset = CustomDataset(X_train, Z_Y_train, Z_train)
        if batch_size == 'full':
            batch_size_ = X_train.shape[0]
        elif isinstance(batch_size, int):
            batch_size_ = batch_size
        data_loader = DataLoader(custom_dataset, batch_size=batch_size_, shuffle=True)
        loss_function = nn.CrossEntropyLoss()

    with tqdm(range(n_epochs)) as epochs:
        epochs.set_description(f"Training the classifier with dataset: {dataset_name}, seed: {seed}")
        for epoch in epochs:
            net.train()
            for i, (covariate_batch, y_batch, z_batch) in enumerate(data_loader):
                covariate_batch, y_batch, z_batch = covariate_batch.to(device), y_batch.to(device), z_batch.to(device)
                Yhat = net(covariate_batch)

                # prediction loss
                cost = loss_function(Yhat.squeeze(), y_batch)

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()


                epochs.set_postfix(loss=cost.item())
            if dataset_name == 'AdultCensus':
                lr_schedule.step()


    if not blind:
        eta_train = net(XZ_train).detach().cpu().numpy().squeeze()
        eta_test = net(XZ_test).detach().cpu().numpy().squeeze()
        Y_hat_train = (eta_train>0.5)
        Y_hat_test = (eta_test>0.5)
    elif blind:
        eta_postproc = net(X_postproc).detach().cpu().numpy().squeeze()
        pred_Z_postproc = eta_postproc[:, 2] + eta_postproc[:, 3]
        eta_postproc = eta_postproc[:, 1] + eta_postproc[:, 3]
        Y_hat_postproc = (eta_postproc>0.5)
        
        eta_test = net(X_test).detach().cpu().numpy().squeeze()
        pred_Z_test = eta_test[:, 2] + eta_test[:, 3]
        eta_test = eta_test[:, 1] + eta_test[:, 3]
        Y_hat_test = (eta_test>0.5)

    pa = torch.mean(Z_train).item()

    df_test = pd.DataFrame()

    level_list = get_level(dataset_name)
    for level in level_list:
        start_time = thread_cpu_time()
        if not blind:
            t = threshold_flipping(pa,eta_train, Y_hat_train,Y_train_np,Z_train_np,level)
            Y_hat_test_new =  postprocessing_flipping(pa,eta_test,Y_hat_test,Z_test_np,t)
        elif blind:
            t = threshold_flipping(pa,eta_postproc, Y_hat_postproc,Y_postproc.numpy(),pred_Z_postproc,level)
            Y_hat_test_new =  postprocessing_flipping(pa,eta_test,Y_hat_test,pred_Z_test,t)
        acc = cal_acc_PPF(Y_hat_test_new,Y_test_np)
        disparity = cal_disparity_PPF(Y_hat_test_new,Z_test_np)
        data = [seed,dataset_name,level,acc, np.abs(disparity), thread_cpu_time()-start_time]
        columns = ['seed','dataset','level','acc', 'disparity', 'time']

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

def training_PPF(dataset_name,blind,seed):
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
        # Create a classifier model
        net = Classifier(n_inputs=input_dim)
    elif blind:
        input_dim = dataset.X_train.shape[1]
        # Create a classifier model
        net = multi_Classifier(n_inputs=input_dim, n_classes=4)
    
    net = net.to(device)
    n_epochs, lr, batch_size= get_training_parameters(dataset_name)

    # Set an optimizer
    lr_decay = 0.98
    optimizer = optim.Adam(net.parameters(), lr=lr)
    lr_schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    # Fair classifier training
    Result = PPF(dataset=dataset,dataset_name=dataset_name,blind=blind,
                     net=net,
                     optimizer=optimizer,lr_schedule=lr_schedule,
                     device=device, n_epochs=n_epochs, batch_size=batch_size, seed=seed)
    print(Result)

    if not blind:
        Result.to_csv(f'Result/PPF/NNo/result_of_{dataset_name}_with_seed_{seed}')
    elif blind:
        Result.to_csv(f'Result/PPF/NNo/result_of_{dataset_name}_with_seed_{seed}_blind')


