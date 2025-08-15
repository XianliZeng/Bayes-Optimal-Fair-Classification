import numpy as np
import torch
import pandas as pd
import random
import time
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from fairlearn.reductions import DemographicParity, ExponentiatedGradient
#from aif360.sklearn.inprocessing import ExponentiatedGradientReduction
from sklearn.ensemble import HistGradientBoostingClassifier

from DataLoader.dataloader import FairnessDataset
from utils import cal_acc, cal_disparity
import resource

def thread_cpu_time():
    r = resource.getrusage(resource.RUSAGE_THREAD)
    return r.ru_utime + r.ru_stime

# ─── HYPERPARAMETERS PER DATASET ─────────────────────────────────────────────

def get_training_parameters(dataset_name):
    if dataset_name == 'COMPAS':
        return 100, 1e-3, 512, 0.2
    elif dataset_name == 'AdultCensus':
        return 200, 1e-1, 512, 0.01
    elif dataset_name == 'Lawschool':
        return 200, 2e-4, 2048, 0.1
    elif dataset_name == 'ACSIncome':
        return 20, 1e-3, 128, 0.01
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# ─── MLP CLASSIFIER DEFINITION ───────────────────────────────────────────────

class MLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classes=2, hidden_layer_sizes=(32, 32, 32),
                 n_epochs=20, batch_size=128, lr=1e-3, gamma=0.0, random_state=0, device='cpu'):
        self.n_classes = n_classes
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.random_state = random_state
        self.device = device

        self.model = None
        self._is_fitted = False

    def _to_numpy(self, X):
        return X.values if isinstance(X, pd.DataFrame) else X

    def _build_model(self, input_dim):
        layers = []
        dims = [input_dim] + list(self.hidden_layer_sizes) + [self.n_classes]
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        return nn.Sequential(*layers)

    def fit(self, X, y, sample_weight=None):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_np = self._to_numpy(X)
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = self._build_model(X_np.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.gamma)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(self.n_epochs):
            self.model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                out = self.model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                optimizer.step()

        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("You must train the classifier before predicting!")
        X_np = self._to_numpy(X)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_np, dtype=torch.float32).to(self.device)
            logits = self.model(X_tensor)
            preds = torch.argmax(logits, dim=1)
        return preds.cpu().numpy()

    def predict_proba(self, X):
        if not self._is_fitted:
            raise RuntimeError("You must train the classifier before predicting!")
        X_np = self._to_numpy(X)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_np, dtype=torch.float32).to(self.device)
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

# ─── TRAINING + EVALUATION ───────────────────────────────────────────────────

def training_RED(
    dataset_name,
    epsilon=0.10,
    max_iter=50,
    seed=0,
    model="mlp",  # add model selector: "mlp" or "lgbm"
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load dataset
    ds = FairnessDataset(dataset=dataset_name, seed=seed, device='cpu')
    ds.normalize()
    (X_tr, Y_tr, Z_tr, _), (X_te, Y_te, Z_te, _) = ds.get_dataset_in_tensor()

    X_train = X_tr.cpu().numpy()
    y_train = Y_tr.squeeze(-1).cpu().numpy().astype(int)
    A_train = Z_tr.squeeze(-1).cpu().numpy().astype(int)
    XZ_train = np.column_stack([A_train, X_train])
    
    X_test = X_te.cpu().numpy()
    y_test = Y_te.squeeze(-1).cpu().numpy().astype(int)
    A_test = Z_te.squeeze(-1).cpu().numpy().astype(int)
    XZ_test  = np.column_stack([A_test, X_test])

    n_epochs, lr, batch_size, gamma = get_training_parameters(dataset_name)
    n_classes = len(np.unique(y_train))

    print(f"\n=== REDUCTION ({model.upper()}+FAIRLEARN) ON {dataset_name} ε={epsilon} max_iter={max_iter} ===")
    start_time = time.time()

    if model == "mlp":
        clf = MLPClassifier(
            n_classes=n_classes,
            hidden_layer_sizes=(500, 200, 100),
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            gamma=gamma,
            random_state=seed,
            device='cpu'
        )
    elif model == "lgbm":
        clf = LGBMClassifier(
            random_state=seed,
            verbosity=-1,
            learning_rate=0.1,
            n_estimators=50,
            max_depth=3,
            num_leaves=7,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8
        )
    elif model == "hgb":
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_depth=4,
            l2_regularization=1.0,
            random_state=seed
        )
    else:
        raise ValueError(f"Unsupported model type: {model}")

    constraint = DemographicParity(difference_bound=epsilon)
    mitigator = ExponentiatedGradient(
        estimator=clf,
        constraints=constraint,
        eps=epsilon,
        max_iter=max_iter
    )
    mitigator.fit(XZ_train, y_train, sensitive_features=A_train)

    y_probs = mitigator._pmf_predict(XZ_test)
    y_pred = np.argmax(y_probs, axis=1)

    elapsed_time = time.time() - start_time

    acc = cal_acc(y_pred, y_test, A_test, 0.5, 0.5)
    disp = cal_disparity(y_pred, A_test, 0.5, 0.5)

    df = pd.DataFrame([[seed, dataset_name, epsilon, acc, abs(disp), elapsed_time]],
                      columns=['seed', 'dataset', 'epsilon', 'acc', 'disparity', 'time'])
    print(df)

    df.to_csv(f"Result/RED/{model.upper()}/result_of_{dataset_name}_with_seed_{seed}_para_{int(epsilon*1000)}",
              index=False)
    return df
