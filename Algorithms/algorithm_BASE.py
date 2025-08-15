import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from DataLoader.dataloader import FairnessDataset, CustomDataset
import time
import resource
from utils import cal_disparity

# Define the network
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
        raise ValueError("Unknown dataset name")

def thread_cpu_time():
    r = resource.getrusage(resource.RUSAGE_THREAD)
    return r.ru_utime + r.ru_stime

def training_BASE(dataset_name, blind=False, seed=0):
    print(f"Running baseline training on {dataset_name} with seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cpu')

    start_time = thread_cpu_time()

    dataset = FairnessDataset(dataset=dataset_name, seed=seed, device=device)
    dataset.normalize()

    training_tensors, testing_tensors = dataset.get_dataset_in_tensor()
    X_train, Y_train, Z_train, XZ_train = training_tensors
    X_test, Y_test, Z_test, XZ_test = testing_tensors

    n_epochs, lr, batch_size = get_training_parameters(dataset_name)

    if not blind:
        custom_dataset = CustomDataset(XZ_train, Y_train, Z_train)
        if batch_size == 'full':
            batch_size_ = XZ_train.shape[0]
        elif isinstance(batch_size, int):
            batch_size_ = batch_size
        input_dim = XZ_train.shape[1]
        data_loader = DataLoader(custom_dataset, batch_size=batch_size_, shuffle=True)
    elif blind:
        custom_dataset = CustomDataset(X_train, Y_train, Z_train)
        if batch_size == 'full':
            batch_size_ = X_train.shape[0]
        elif isinstance(batch_size, int):
            batch_size_ = batch_size
        input_dim = X_train.shape[1]
        data_loader = DataLoader(custom_dataset, batch_size=batch_size_, shuffle=True)

    net = Classifier(n_inputs=input_dim).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    loss_fn = nn.BCELoss()

    for epoch in tqdm(range(n_epochs), desc="Training"):
        net.train()
        for i, (xb, yb, zb) in enumerate(data_loader):
            xb, yb = xb.to(device), yb.to(device)
            preds = net(xb).squeeze()
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if dataset_name == 'AdultCensus':
            scheduler.step()

    # Evaluation
    net.eval()
    with torch.no_grad():
        if not blind:
          preds = net(XZ_test).squeeze().cpu().numpy()
        elif blind:
          preds = net(X_test).squeeze().cpu().numpy()
        preds_binary = (preds > 0.5).astype(int)
        acc = np.mean(preds_binary == Y_test.cpu().numpy())
        disparity = cal_disparity(preds, Z_test.cpu().numpy(), 0.5, 0.5)

    elapsed_time = thread_cpu_time() - start_time

    data = [seed, dataset_name, 0.0, acc, np.abs(disparity), elapsed_time]
    columns = ['seed', 'dataset', 'level', 'acc', 'disparity', 'time']
    df_result = pd.DataFrame([data], columns=columns)

    if not blind:
        filename = f'Result/BASE/NNo/result_of_{dataset_name}_with_seed_{seed}'
    else:
        filename = f'Result/BASE/NNo/result_of_{dataset_name}_with_seed_{seed}_blind'

    df_result.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

