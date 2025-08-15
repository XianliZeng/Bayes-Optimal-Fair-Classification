import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from BlackBoxAuditing.repairers.GeneralRepairer import Repairer
from DataLoader.dataloader_DIR import Get_data_tensor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from DataLoader.dataloader_DIR import CustomDataset, FairnessDataset

from utils import cal_acc, cal_disparity
import resource

import torch.optim as optim


def thread_cpu_time():
    r = resource.getrusage(resource.RUSAGE_THREAD)
    return r.ru_utime + r.ru_stime


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
        return self.model(x)


def DIR(dataset, dataset_name, blind, net, optimizer, lr_schedule,
        level, device, n_epochs=200, batch_size=2048, seed=0):

    training_set, testing_set = dataset.get_dataset()

    Y_train = training_set['Target']
    XZ_train_df = training_set.drop('Target', axis=1)
    index_train = XZ_train_df.keys().tolist().index('Sensitive')
    features_train = XZ_train_df.values.tolist()

    Y_test = testing_set['Target']
    XZ_test_df = testing_set.drop('Target', axis=1)
    index_test = XZ_test_df.keys().tolist().index('Sensitive')
    features_test = XZ_test_df.values.tolist()

    repairer_train = Repairer(features_train, index_train, level, False)
    repairer_test = Repairer(features_test, index_test, level, False)

    data_train = np.array(repairer_train.repair(features_train))
    data_test  = np.array(repairer_test.repair(features_test))
    
    if data_train.ndim == 1:
        data_train = data_train.reshape(-1, XZ_train_df.shape[1])
    if data_test.ndim == 1:
        data_test = data_test.reshape(-1, XZ_test_df.shape[1])

    keys    = XZ_train_df.keys()
    df_train = pd.DataFrame(data=data_train, columns=keys)
    df_test  = pd.DataFrame(data=data_test,  columns=keys)

    Y_train = Y_train.reset_index(drop=True)
    Y_test = Y_test.reset_index(drop=True)

    training_tensors, testing_tensors = Get_data_tensor(
        dataset_name, df_train, df_test, Y_train, Y_test, device
    )
    X_train, Y_train_t, Z_train, XZ_train = training_tensors
    X_test,  Y_test_t,  Z_test,  XZ_test  = testing_tensors

    if not blind:
        ds = CustomDataset(XZ_train, Y_train_t, Z_train)
        bs = XZ_train.shape[0] if batch_size == 'full' else batch_size
        data_loader = DataLoader(ds, batch_size=bs, shuffle=True)
    else:
        ds = CustomDataset(X_train, Y_train_t, Z_train)
        bs = X_train.shape[0] if batch_size == 'full' else batch_size
        data_loader = DataLoader(ds, batch_size=bs, shuffle=True)

    loss_function = nn.BCELoss()
    start_time = thread_cpu_time()

    with tqdm(range(n_epochs)) as epochs:
        desc = f"Training classifier: {dataset_name}, seed {seed}, level {level}"
        epochs.set_description(desc)
        for epoch in epochs:
            net.train()
            for cov_batch, y_batch, z_batch in data_loader:
                cov_batch = cov_batch.to(device)
                y_batch   = y_batch.to(device)
                z_batch   = z_batch.to(device)

                Yhat = net(cov_batch)
                cost = loss_function(Yhat.squeeze(), y_batch)

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                epochs.set_postfix(loss=cost.item())

            if dataset_name == 'AdultCensus':
                lr_schedule.step()

    net.eval()
    with torch.no_grad():
        if not blind:
            eta_test = net(XZ_test).cpu().numpy().squeeze()
        else:
            eta_test = net(X_test).cpu().numpy().squeeze()

    Y_test_np = Y_test_t.cpu().numpy() if isinstance(Y_test_t, torch.Tensor) else Y_test_t.to_numpy()
    Z_test_np = Z_test.cpu().numpy()

    acc = cal_acc(eta_test, Y_test_np, Z_test_np, 0.5, 0.5)
    disparity = cal_disparity(eta_test, Z_test_np, 0.5, 0.5)

    elapsed = thread_cpu_time() - start_time
    data = [seed, dataset_name, level, acc, abs(disparity), elapsed]
    columns = ['seed', 'dataset', 'level', 'acc', 'disparity', 'time']
    df_result = pd.DataFrame([data], columns=columns)

    return df_result


def get_training_parameters(dataset_name):
    if dataset_name == 'AdultCensus':
        return 200, 1e-1, 512
    elif dataset_name == 'COMPAS':
        return 500, 5e-4, 2048
    elif dataset_name == 'Lawschool':
        return 200, 2e-4, 2048
    elif dataset_name == 'ACSIncome':
        return 20, 1e-3, 128
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")


def training_DIR(dataset_name, level, blind, seed):
    device = torch.device('cpu')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load the raw DataFrame without any repair
    dataset = FairnessDataset(dataset=dataset_name, seed=seed, device=device)
    T1, T2 = dataset.get_dataset()

    # Hyperparameters
    n_epochs, lr, batch_size = get_training_parameters(dataset_name)

    if not blind:
        input_dim = len(T1.keys()) - 1
    else:
        input_dim = len(T1.keys()) - 2 

    # Instantiate Classifier with that input_dim
    net = Classifier(n_inputs=input_dim).to(device)

    # Now optimizer and scheduler use the correctly sized net
    optimizer = optim.Adam(net.parameters(), lr=lr)
    lr_schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # Pass this net into DIR (which will train it on the repaired data)
    result_df = DIR(
        dataset=dataset,
        dataset_name=dataset_name,
        blind=blind,
        net=net,
        optimizer=optimizer,
        lr_schedule=lr_schedule,
        level=level,
        device=device,
        n_epochs=n_epochs,
        batch_size=batch_size,
        seed=seed
    )

    print(result_df)
    suffix = "" if not blind else "_blind"
    result_df.to_csv(
        f"Result/DIR/NNo/result_of_{dataset_name}_with_seed_{seed}_para_{int(level*1000)}{suffix}",
        index=False
    )