import time
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin

from DataLoader.dataloader import FairnessDataset
from utils import cal_acc, cal_disparity
# Necessary functions + hyperparameters for KDE method (Cho et al. 2020)
from Algorithms.algorithm_KDE import CDF_tau, Huber_loss

# global tau and Q_function (if needed)
tau = 0.5
# López-Benítez & Casadevall (2011) approximation of Q-function
a, b, c = 0.4920, 0.2887, 1.1893
Q_function = lambda x: torch.exp(-a * x**2 - b * x - c)

import resource

def thread_cpu_time():
    r = resource.getrusage(resource.RUSAGE_THREAD)
    return r.ru_utime + r.ru_stime

def get_level(dataset_name):
    if dataset_name == 'AdultCensus':
        return np.array([0, 1, 2, 4, 6, 10, 15, 25, 35, 50])
    if dataset_name == 'COMPAS':
        return np.array([0, 1, 2, 4, 6, 10, 15, 25, 35, 50])
    if dataset_name == 'Lawschool':
        return np.array([0, 1, 2, 4, 6, 10, 15, 25, 35, 50])
    if dataset_name == 'ACSIncome':
        return np.array([0, 1, 2, 4, 6, 10, 15, 25, 35, 50])
    raise ValueError(f"Unknown dataset: {dataset_name}")

def get_training_parameters(dataset_name):
    if dataset_name == 'AdultCensus':  return 200, 1e-1, 512
    if dataset_name == 'COMPAS':      return 500, 5e-4, 2048
    if dataset_name == 'Lawschool':    return 200, 2e-4, 2048
    if dataset_name == 'ACSIncome':    return 20, 1e-3, 128
    raise ValueError(f"Unknown dataset: {dataset_name}")

def default_model_fn(input_dim, hidden_dims=[500,200,100], output_dim=1):
    layers=[]
    dims=[input_dim]+hidden_dims
    for in_d,out_d in zip(dims[:-1],dims[1:]):
        layers.append(nn.Linear(in_d,out_d)); layers.append(nn.ReLU())
    layers.append(nn.Linear(dims[-1],output_dim)); layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)

def single_layer_model_fn(input_dim, hidden_dims=None, output_dim=1):
    if hidden_dims is None:
        hidden_dims=[128]
    layers=[]
    dims=[input_dim]+hidden_dims
    for in_d,out_d in zip(dims[:-1],dims[1:]):
        layers.append(nn.Linear(in_d,out_d)); layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dims[-1], output_dim))  # raw logit correction
    return nn.Sequential(*layers)

# RBF-MMD used for in-processing only
def rbf_mmd(x, y, gamma=None, eps=1e-12):
    if gamma is None:
        gamma = 1.0/(0.1 + eps)
    xy = torch.cat([x, y], dim=0)
    D2 = (xy.unsqueeze(1) - xy.unsqueeze(0)).pow(2).sum(-1)
    K = torch.exp(-gamma * D2)
    n, m = x.size(0), y.size(0)
    Kxx = K[:n,:n] - torch.eye(n, device=K.device)*K[:n,:n]
    Kyy = K[n:,n:] - torch.eye(m, device=K.device)*K[n:,n:]
    Kxy = K[:n,n:]
    return Kxx.sum()/(n*(n-1)) + Kyy.sum()/(m*(m-1)) - 2*Kxy.sum()/(n*m)

class TorchNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model_fn, optimizer_fn,
                 loss_fn=nn.BCELoss, epochs=10, batch_size=None, device='cpu'):
        self.model_fn, self.optimizer_fn = model_fn, optimizer_fn
        self.loss_fn = loss_fn
        self.epochs, self.batch_size, self.device = epochs, batch_size, device

    def fit(self, X, y, sample_weight=None):
        self.model = self.model_fn(X.shape[1]).to(self.device)
        self.optimizer = self.optimizer_fn(self.model.parameters())
        X_t = torch.tensor(X.astype(np.float32), device=self.device)
        y_t = torch.tensor(y.astype(np.float32).reshape(-1,1), device=self.device)
        w_t = torch.tensor(
            np.ones_like(y_t) if sample_weight is None else sample_weight,
            device=self.device
        )
        loader = DataLoader(
            TensorDataset(X_t, y_t, w_t),
            batch_size=self.batch_size or len(X_t), shuffle=True
        )
        criterion = self.loss_fn(reduction='none')
        self.model.train()
        for _ in tqdm(range(self.epochs), desc='Base training'):
            for xb, yb, wb in loader:
                preds = self.model(xb)
                loss = (criterion(preds, yb) * wb).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_t = torch.tensor(X.astype(np.float32), device=self.device)
        with torch.no_grad():
            probs = self.model(X_t).cpu().numpy().flatten()
        return np.vstack([1-probs, probs]).T

class KDEPostProcessor(BaseEstimator, ClassifierMixin):
    """FRAPPE post-processor using KDE-based fairness regularizer"""
    def __init__(
        self, model_fn=single_layer_model_fn,
        optimizer_fn=lambda p: torch.optim.Adagrad(p, lr=0.01),
        lambda_kde=0.0, h=0.1, delta=1.0,
        epochs=100, batch_size=32, device='cpu'
    ):
        self.model_fn = model_fn
        self.optimizer_fn = optimizer_fn
        self.lambda_kde = lambda_kde
        self.h = h
        self.delta = delta
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

    def fit(self, X, a, base_scores):
        X_arr = np.array(X, dtype=np.float32)
        a_arr = np.array(a, dtype=np.int64)
        s_arr = np.array(base_scores, dtype=np.float32).reshape(-1,1)

        self.correction = self.model_fn(X_arr.shape[1]).to(self.device)
        self.optimizer = self.optimizer_fn(self.correction.parameters())

        X_t = torch.tensor(X_arr, device=self.device)
        a_t = torch.tensor(a_arr, device=self.device)
        s_t = torch.tensor(s_arr, device=self.device)

        loader = DataLoader(
            TensorDataset(X_t, a_t, s_t),
            batch_size=self.batch_size, shuffle=True
        )

        self.correction.train()
        for _ in tqdm(range(self.epochs), desc='KDE PostProc'):
            for xb, ab, sb in loader:
                # Base logits + correction
                logit_base = torch.log(sb / (1-sb) + 1e-6)
                delta = self.correction(xb)
                p_corr = torch.sigmoid(logit_base + delta)
                p_base = sb.clamp(1e-6, 1-1e-6)

                # Distillation KL loss
                d_pred = (
                    p_base * torch.log(p_base / p_corr)
                    + (1-p_base) * torch.log((1-p_base) / (1-p_corr))
                ).mean()

                # KDE-based fairness penalty
                c_all = CDF_tau(p_corr, h=self.h, tau=tau)
                c0 = CDF_tau(p_corr[ab==0], h=self.h, tau=tau)
                c1 = CDF_tau(p_corr[ab==1], h=self.h, tau=tau)
                loss_kde = (
                    Huber_loss(c0 - c_all, self.delta)
                    + Huber_loss(c1 - c_all, self.delta)
                )

                # Total loss
                loss = d_pred + self.lambda_kde * loss_kde
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self

    def predict(self, X, base_scores, threshold=0.5):
        X_arr = np.array(X, dtype=np.float32)
        s_arr = np.array(base_scores, dtype=np.float32).reshape(-1,1)
        X_t = torch.tensor(X_arr, device=self.device)
        s_t = torch.tensor(s_arr, device=self.device)

        self.correction.eval()
        with torch.no_grad():
            delta = self.correction(X_t)
            p_corr = torch.sigmoid(torch.log(s_t/(1-s_t) + 1e-6) + delta)
        return (p_corr.cpu().numpy().flatten() >= threshold).astype(int)


def training_FRAPPE(dataset_name, lambda_mmd, seed, in_processing=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')

    # Load and normalize dataset
    ds = FairnessDataset(dataset=dataset_name, seed=seed, device=device)
    ds.normalize()
    (X_tr, Y_tr, Z_tr, XZ_tr), (X_te, Y_te, Z_te, XZ_te) = ds.get_dataset_in_tensor()
    X_train, Y_train, Z_train = XZ_tr.cpu().numpy(), Y_tr.cpu().numpy(), Z_tr.cpu().numpy()
    X_test,  Y_test,  Z_test  = XZ_te.cpu().numpy(), Y_te.cpu().numpy(),  Z_te.cpu().numpy()

    epochs, lr, bs = get_training_parameters(dataset_name)

    if in_processing:
        start_time = thread_cpu_time()
        model = default_model_fn(input_dim=X_train.shape[1], hidden_dims=[32,32,32]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        dataset = TensorDataset(torch.tensor(X_train, device=device), torch.tensor(Z_train, device=device), torch.tensor(Y_train, device=device))
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)
        df = pd.DataFrame()

        model.train()
        for epoch in tqdm(range(epochs), desc='InProc training'):
            for xb, ab, yb in loader:
                preds = model(xb)
                loss_primary = criterion(preds, yb.unsqueeze(1))
                p0 = preds[ab==0]; p1 = preds[ab==1]
                mmd = rbf_mmd(p0, p1) if p0.numel()>=2 and p1.numel()>=2 else torch.tensor(0.0, device=device)
                loss = loss_primary + lambda_mmd * mmd
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): probs = model(torch.tensor(X_test, device=device)).cpu().numpy().flatten()
        Y_pred = (probs>=0.5).astype(int); final_proba=probs

        acc = cal_acc(Y_pred, Y_test, Z_test, 0.5, 0.5)
        disparity = cal_disparity(Y_pred, Z_test, 0.5, 0.5)

        data =[seed, dataset_name, lambda_mmd, in_processing, acc, np.abs(disparity), thread_cpu_time()-start_time]
        columns = ['seed','dataset','lambda_mmd','in_processing','acc','disparity','time']

        temp_dp = pd.DataFrame([data], columns=columns)
        df= pd.concat([df,temp_dp])
        print(df)
        fname = f"Result/MinDiff/NNo/result_of_{dataset_name}_with_seed_{seed}_para_{int(lambda_mmd*1000)}"
        df.to_csv(fname, index=False)
    else:
        # FRAPPE post-process on top of KDE
        # 1) Train base model
        if dataset_name == 'ACSIncome':
            model_fn = lambda input_dim: default_model_fn(input_dim=input_dim, hidden_dims=[500,200,100])
            print("Running on ACSIncome...")
        else:
            model_fn = lambda input_dim: default_model_fn(input_dim=input_dim, hidden_dims=[32,32,32])
        base = TorchNNClassifier(
            model_fn=model_fn,
            optimizer_fn=lambda p: torch.optim.Adam(p, lr=lr),
            epochs=epochs,
            batch_size=bs,
            device=device
        )
        base.fit(X_train, Y_train)
        base_scores_train = base.predict_proba(X_train)[:,1]
        base_scores_test  = base.predict_proba(X_test)[:,1]

        # 2) Post-process with KDE for each lambda
        results = []
        for level in get_level(dataset_name):
            start_time = thread_cpu_time()
            post = KDEPostProcessor(
                model_fn=single_layer_model_fn,
                optimizer_fn=lambda p: torch.optim.Adagrad(p, lr=0.01),
                lambda_kde=level,
                h=0.1,
                delta=1.0,
                epochs=int(epochs/5),
                batch_size=bs,
                device=device
            )
            post.fit(X_train, Z_train, base_scores_train)
            Y_pred = post.predict(X_test, base_scores_test)
            acc = cal_acc(Y_pred, Y_test, Z_test, 0.5, 0.5)
            disp = cal_disparity(Y_pred, Z_test, 0.5, 0.5)
            results.append([seed, dataset_name, level, acc, abs(disp), thread_cpu_time() - start_time])

        df = pd.DataFrame(
            results,
            columns=['seed','dataset','level','acc','disparity','time']
        )
        print(df)
        df.to_csv(f"Result/KDE_FRAPPE/NNo/result_of_{dataset_name}_with_seed_{seed}", index=False)

    return