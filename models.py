import torch
import torch.nn as nn


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


class domain_Classifier(nn.Module):
    def __init__(self):
        super(domain_Classifier, self).__init__()
        self.model = nn.Sequential(
                        nn.Linear(1, 1),
                        nn.Sigmoid()
                        )

    def forward(self,x):
        pred = self.model(x)
        return pred

class multi_Classifier(nn.Module):
    def __init__(self, n_inputs, n_classes):
        super(multi_Classifier, self).__init__()
        self.model = nn.Sequential(
                        nn.Linear(n_inputs, 32),
                        nn.ReLU(),
                        nn.Linear(32, 32),
                        nn.ReLU(),
                        nn.Linear(32, n_classes)
                    )

    def forward(self, x):
        predict = self.model(x)
        return predict