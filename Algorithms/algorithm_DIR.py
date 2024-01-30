import random
import IPython
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from BlackBoxAuditing.repairers.GeneralRepairer import Repairer
from DataLoader.dataloader_RDI import Get_data_tensor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from DataLoader.dataloader_RDI import CustomDataset
from DataLoader.dataloader_RDI import FairnessDataset

from utils import cal_acc, cal_disparity
import time


import torch.optim as optim
from models import Classifier

def RDI(dataset,dataset_name, net, optimizer,lr_schedule, level, device, n_epochs=200, batch_size=2048, seed=0):






    # Retrieve train/test splitted pytorch tensors for index=split
    training_set, testing_set = dataset.get_dataset()

    Y_train = training_set['Target']
    XZ_train = training_set.drop('Target', axis=1)
    index_train = XZ_train.keys().tolist().index('Sensitive')
    features_train = XZ_train.values.tolist()


    Y_test = testing_set['Target']
    XZ_test = testing_set.drop('Target', axis=1)
    index_test = XZ_test.keys().tolist().index('Sensitive')
    features_test = XZ_test.values.tolist()



    repairer_train = Repairer(features_train, index_train, level, False)
    repairer_test = Repairer(features_test, index_test, level, False)

    data_train = np.array(repairer_train.repair(features_train))
    data_test = np.array(repairer_test.repair(features_test))


    keys = XZ_train.keys()




    df_train = pd.DataFrame(data=data_train, columns=keys)
    df_test = pd.DataFrame(data=data_test, columns=keys)

    training_tensors,  testing_tensors = Get_data_tensor(dataset_name, df_train,  df_test, Y_train,  Y_test, device)



    X_train, Y_train, Z_train, XZ_train = training_tensors
    X_test, Y_test, Z_test, XZ_test = testing_tensors

    # training data size and validation data size



    custom_dataset = CustomDataset(XZ_train, Y_train, Z_train)
    if batch_size == 'full':
        batch_size_ = XZ_train.shape[0]
    elif isinstance(batch_size, int):
        batch_size_ = batch_size
    data_loader = DataLoader(custom_dataset, batch_size=batch_size_, shuffle=True)



    loss_function = nn.BCELoss()
    with tqdm(range(n_epochs)) as epochs:
        epochs.set_description(f"Training the classifier with dataset: {dataset_name}, seed: {seed}, level: {level}")
        for epoch in epochs:
            net.train()
            for i, (xz_batch, y_batch, z_batch) in enumerate(data_loader):
                xz_batch, y_batch, z_batch = xz_batch.to(device), y_batch.to(device), z_batch.to(device)
                Yhat = net(xz_batch)

                # prediction loss
                cost = loss_function(Yhat.squeeze(), y_batch)

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()


                epochs.set_postfix(loss=cost.item())
            if dataset_name == 'AdultCensus':
                lr_schedule.step()

    eta_test = net(XZ_test).detach().cpu().numpy().squeeze()
    Y_test_np = Y_test.clone().detach().numpy()
    Z_test_np = Z_test.clone().detach().numpy()

    acc = cal_acc(eta_test,Y_test_np,Z_test_np,0.5,0.5)
    disparity = cal_disparity(eta_test,Z_test_np,0.5,0.5)

    data = [seed,dataset_name,level,acc, np.abs(disparity)]
    columns = ['seed','dataset','level','acc', 'disparity']


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
    return n_epochs,lr,batch_size






def training_DIR(dataset_name, level,seed):
    device = torch.device('cpu')


    # Set a seed for random number generation
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Import dataset
    dataset = FairnessDataset(dataset=dataset_name, seed = seed, device=device)
    T1,  T2= dataset.get_dataset()
    input_dim = len(T1.keys())-1
    n_epochs, lr, batch_size= get_training_parameters(dataset_name)
    # Create a classifier model

    net = Classifier(n_inputs=input_dim)
    net = net.to(device)

    # Set an optimizer
    lr_decay = 0.98
    optimizer = optim.Adam(net.parameters(), lr=lr)
    lr_schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    # Fair classifier training


    Result = RDI(dataset=dataset,dataset_name=dataset_name,
                 net=net, optimizer=optimizer, lr_schedule=lr_schedule,level = level,
                 device=device, n_epochs=n_epochs, batch_size=batch_size, seed=seed)

    print(Result)
    Result.to_csv(f'Result/DIR/result_of_{dataset_name}_with_seed_{seed}_para_{int(level*1000)}')










