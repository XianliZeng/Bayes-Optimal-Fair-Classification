import random
import numpy as np
import pandas as pd
from DataLoader.dataloader import CustomDataset
from utils import  cal_acc,cal_disparity
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import time
#from models import Classifier
from DataLoader.dataloader import FairnessDataset

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

tau = 0.5
# Approximation of Q-function given by López-Benítez & Casadevall (2011) based on a second-order exponential function & Q(x) = 1- Q(-x):
a = 0.4920
b = 0.2887
c = 1.1893
Q_function = lambda x: torch.exp(-a*x**2 - b*x - c) 

def CDF_tau(Yhat, h=0.01, tau=0.5):
    m = len(Yhat)
    Y_tilde = (tau-Yhat)/h
    sum_ = torch.sum(Q_function(Y_tilde[Y_tilde>0])) \
           + torch.sum(1-Q_function(torch.abs(Y_tilde[Y_tilde<0]))) \
           + 0.5*(len(Y_tilde[Y_tilde==0]))
    return sum_/m

def Huber_loss(x, delta):
    if x.abs() < delta:
        return (x ** 2) / 2
    return delta * (x.abs() - delta / 2)

def KDE(dataset,dataset_name, blind, net, optimizer,lr_schedule, lambda_, h, delta, device, n_epochs=200, batch_size=2048, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    start_time = thread_cpu_time()
    training_tensors,  testing_tensors = dataset.get_dataset_in_tensor()
    X_train, Y_train, Z_train, XZ_train = training_tensors
    X_test, Y_test, Z_test, XZ_test = testing_tensors

    Z_train_np = Z_train.clone().detach().cpu().numpy()

    if Z_train_np.mean()==0 or Z_train_np.mean()==1:
        print('At least one sensitive group has no data point')
        sys.exit()


    sensitive_attrs = dataset.sensitive_attrs
    if not blind:
        custom_dataset = CustomDataset(XZ_train, Y_train, Z_train)
        if batch_size == 'full':
            batch_size_ = XZ_train.shape[0]
        elif isinstance(batch_size, int):
            batch_size_ = batch_size
        data_loader = DataLoader(custom_dataset, batch_size=batch_size_, shuffle=True)
    if blind:
        custom_dataset = CustomDataset(X_train, Y_train, Z_train)
        if batch_size == 'full':
            batch_size_ = X_train.shape[0]
        elif isinstance(batch_size, int):
            batch_size_ = batch_size
        data_loader = DataLoader(custom_dataset, batch_size=batch_size_, shuffle=True)
    
    pi = torch.tensor(np.pi).to(device)
    phi = lambda x: torch.exp(-0.5*x**2)/torch.sqrt(2*pi) #normal distribution
    
    # An empty dataframe for logging experimental results

    loss_function = nn.BCELoss()
    costs = []
    with tqdm(range(n_epochs)) as epochs:
        epochs.set_description(f"Training the classifier with seed: dataset: {dataset_name}, {seed}, lambda: {lambda_}")
        for epoch in epochs:
            net.train()

            for i, (covariate_batch, y_batch, z_batch) in enumerate(data_loader):
                covariate_batch, y_batch, z_batch = covariate_batch.to(device), y_batch.to(device), z_batch.to(device)
                Yhat = net(covariate_batch)
                cost = 0
                m = z_batch.shape[0]

                # prediction loss
                p_loss = loss_function(Yhat.squeeze(), y_batch)
                cost += (1 - lambda_) * p_loss

                # DP_Constraint
                Pr_Ytilde1 = CDF_tau(Yhat.detach(),h,tau)
                for z in sensitive_attrs:
                    Pr_Ytilde1_Z = CDF_tau(Yhat.detach()[z_batch==z],h,tau)
                    m_z = z_batch[z_batch==z].shape[0]

                    Delta_z = Pr_Ytilde1_Z-Pr_Ytilde1
                    Delta_z_grad = torch.dot(phi((tau-Yhat.detach()[z_batch==z])/h).view(-1),
                                                  Yhat[z_batch==z].view(-1))/h/m_z
                    Delta_z_grad -= torch.dot(phi((tau-Yhat.detach())/h).view(-1),
                                                  Yhat.view(-1))/h/m

                    if Delta_z.abs() >= delta:
                        if Delta_z > 0:
                            Delta_z_grad *= lambda_*delta
                            cost += Delta_z_grad
                        else:
                            Delta_z_grad *= -lambda_*delta
                            cost += Delta_z_grad
                    else:
                        Delta_z_grad *= lambda_*Delta_z
                        cost += Delta_z_grad


                optimizer.zero_grad()
                if (torch.isnan(cost)).any():
                    continue
                cost.backward()
                optimizer.step()
                epochs.set_postfix(loss=cost.item())
            if dataset_name == 'AdultCensus':
                lr_schedule.step()

    if not blind:
        eta_test = net(XZ_test).detach().cpu().numpy().squeeze()
    elif blind:
        eta_test = net(X_test).detach().cpu().numpy().squeeze()


    Y_test_np = Y_test.clone().detach().numpy()
    Z_test_np = Z_test.clone().detach().numpy()

    acc = cal_acc(eta_test,Y_test_np,Z_test_np,0.5,0.5)
    disparity = cal_disparity(eta_test,Z_test_np,0.5,0.5)
    data = [seed,dataset_name,lambda_,acc, np.abs(disparity), thread_cpu_time()-start_time]
    columns = ['seed','dataset','lambda','acc', 'disparity', 'time']
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







def training_KDE(dataset_name,lambda_,blind,seed):

    device = torch.device('cpu')
    h = 0.1
    delta = 1.0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Import dataset

    dataset = FairnessDataset(dataset=dataset_name, seed=seed, device=device)
    dataset.normalize()
    if not blind:
        input_dim = dataset.XZ_train.shape[1]
    elif blind:
        input_dim = dataset.X_train.shape[1]
    n_epochs, lr, batch_size = get_training_parameters(dataset_name)
   # Create a classifier model

    print(f'we are running {dataset_name}_with_seed_{seed} and lambda_{lambda_}')
    net = Classifier(n_inputs=input_dim)
    net = net.to(device)

    # Set an optimizer
    lr_decay = 0.98
    optimizer = optim.Adam(net.parameters(), lr=lr)
    lr_schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    # Fair classifier training
    Result = KDE(dataset=dataset, dataset_name=dataset_name,
                                 blind=blind, net=net,
                                 optimizer=optimizer,lr_schedule=lr_schedule,
                                 lambda_=lambda_, h=h, delta=delta,
                                 device=device, n_epochs=n_epochs, batch_size=batch_size, seed=seed)
    print(Result)

    if not blind:
        Result.to_csv(f'Result/KDE/NNo/result_of_{dataset_name}_with_seed_{seed}_para_{int(lambda_*1000)}')
    elif blind:
        Result.to_csv(f'Result/KDE/NNo/result_of_{dataset_name}_with_seed_{seed}_para_{int(lambda_*1000)}_blind')



