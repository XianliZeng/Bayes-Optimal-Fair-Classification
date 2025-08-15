import time
import random

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from DataLoader.dataloader import FairnessDataset
from utils import cal_acc, cal_disparity

def thread_cpu_time():
     import resource
     r = resource.getrusage(resource.RUSAGE_THREAD)
     return r.ru_utime + r.ru_stime

def get_training_parameters(dataset_name):
    """
    Returns: pretrain_iters, retrain_iters, lr, batch_size, c1
    (c2 unused)
    """
    if dataset_name == 'AdultCensus':
        return 200, 200, 1e-1, 512, 1e-3
    if dataset_name == 'COMPAS':
        return 200, 200, 5e-4, 2048, 1e-3
    if dataset_name == 'Lawschool':
        return 200, 200, 2e-4, 2048, 1e-3
    if dataset_name == 'ACSIncome':
        return 200, 200, 1e-3, 128,  1e-3
    raise ValueError(f"Unknown dataset: {dataset_name}")

def project_simplex(v, s):
    """ O(n log n) projection of v onto { w>=0, sum(w)=s }. """
    v = np.asarray(v, dtype=np.float64)
    n = v.size
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    idx = np.arange(1, n+1)
    ts = (css - s) / idx
    cond = u > ts
    if not np.any(cond):
        return np.ones(n, dtype=v.dtype) * (s / n)
    rho = np.where(cond)[0][-1]
    theta = ts[rho]
    w = v - theta
    return np.maximum(w, 0.0)

def fast_weight_update(losses, gamma, mass=None):
    """
    Projects losses/(2*gamma) onto the simplex of total mass `mass`,
    matching the notebook’s QP (penalty = (γ)*||w||²).
    """
    if mass is None:
        mass = losses.size
    u = losses / (2.0 * gamma)
    return project_simplex(u, s=float(mass))

class AdaptiveReweighingClassifier:
    """
    ARA, mirroring compas.ipynb but with fast projection + sklearn logistic:
      - sum of each group’s w_i = c_max (largest group size)
      - penalty term = (c1+alpha)*||w||^2   ⇒  fast projection uses 1/(2*gamma)
      - base learner = LogisticRegression(penalty='none', warm_start=True)
    """

    def __init__(
        self,
        alpha=0.0,
        c1=1e-3,
        pretrain_iters=200,
        retrain_iters=200,
        tol=1e-3,
        max_outer=25
    ):
        self.alpha = alpha
        self.c1    = c1
        self.pretrain_iters = pretrain_iters
        self.retrain_iters  = retrain_iters
        self.tol   = tol
        self.max_outer = max_outer

        # un-regularized, LBFGS, warm-start logistic
        self.model = LogisticRegression(
            penalty='l2',
            C=1e12,  # effectively no regularization
            solver='lbfgs',
            max_iter=self.pretrain_iters,
            warm_start=True
        )

    def fit(self, X_np, y_np, grp_np):
        n, _ = X_np.shape

        # find c_max = size of largest sensitive group
        unique, counts = np.unique(grp_np, return_counts=True)
        c_max = counts.max()

        # initial uniform weights
        w = np.ones(n)

        # (0) Pretrain on unweighted data
        self.model.max_iter = self.pretrain_iters
        self.model.fit(X_np, y_np)

        for it in range(self.max_outer):
            w_prev = w.copy()

            # (1) surrogate losses = logistic loss
            probs  = self.model.predict_proba(X_np)[:,1]
            losses = -(y_np * np.log(probs + 1e-12)
                       + (1-y_np) * np.log(1-probs + 1e-12))

            # (2) group-wise fast projection
            gamma = self.c1 + self.alpha
            w_new = np.zeros_like(w)
            for g in unique:
                idx      = np.where(grp_np == g)[0]
                losses_g = losses[idx]
                # project onto simplex of mass = c_max
                w_new[idx] = fast_weight_update(losses_g, gamma, mass=c_max)
            w = w_new

            # (3) check relative change
            relchg = np.linalg.norm(w - w_prev) / np.linalg.norm(w_prev)
            print(f"Iter {it+1}, rel-weight-change = {relchg:.4f}")
            if relchg < self.tol:
                print(f"Converged after {it+1} iterations.")
                break

            # (4) retrain weighted logistic
            self.model.max_iter = self.retrain_iters
            self.model.fit(X_np, y_np, sample_weight=w)

        self.sample_weights_ = w
        return self

    def predict(self, X_np, threshold=0.5):
        probs = self.model.predict_proba(X_np)[:,1]
        return (probs >= threshold).astype(int)


def training_ARA(dataset_name, alpha, seed):
    random.seed(seed)
    np.random.seed(seed)

    # load & normalize
    ds = FairnessDataset(dataset=dataset_name, seed=seed, device="cpu")
    ds.normalize()
    (X_tr, Y_tr, Z_tr, XZ_tr), (X_te, Y_te, Z_te, XZ_te) = ds.get_dataset_in_tensor()
    X_train = XZ_tr.cpu().numpy()
    y_train = Y_tr.cpu().numpy()
    z_train = Z_tr.cpu().numpy()
    X_test  = XZ_te.cpu().numpy()

    # hyperparameters
    pre_e, retr_e, lr, bs, c1 = get_training_parameters(dataset_name)

    start_time = thread_cpu_time()
    clf = AdaptiveReweighingClassifier(
        alpha=alpha,
        c1=c1,
        pretrain_iters=pre_e,
        retrain_iters=retr_e,
        tol=1e-3,
        max_outer=25
    )

    clf.fit(X_train, y_train, z_train)
    y_pred = clf.predict(X_test)

    acc       = cal_acc(y_pred, Y_te.cpu().numpy(), Z_te.cpu().numpy(), 0.5, 0.5)
    disparity = cal_disparity(y_pred, Z_te.cpu().numpy(), 0.5, 0.5)

    df = pd.DataFrame([[ 
        seed, dataset_name, alpha,
        acc, abs(disparity),
        thread_cpu_time() - start_time,
    ]], columns=['seed','dataset','alpha','acc','disparity','time'])
    print(df)

    fname = f"Result/ARA/NNo/result_of_{dataset_name}_with_seed_{seed}_para_{int(alpha*1000)}"
    df.to_csv(fname, index=False)
    return df
