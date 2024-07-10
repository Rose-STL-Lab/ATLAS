import numpy as np
import torch
import torch.nn as nn
import tqdm
from abc import ABC, abstractmethod

from utils import mae
import config


class Predictor(ABC):
    optimizer = None

    @abstractmethod
    def run(self, x):
        pass

    def loss(self, y_pred, y_true):
        return mae(y_pred, y_true)

    # some predictors can be given as fixed functions
    def needs_training(self):
        return True


class LocalTrainer:
    def __init__(self, predictor, basis):
        """
            predictor: local_symmetry.py/Predictor
                This corresponds to xi in the propoasl
            basis: lie_basis.py/GroupBasis
                Basis for the discovered symmetry group (somewhat corresponding to G in the proposal)
        """
        self.predictor = predictor
        self.basis = basis
       
        if config.DEBUG:
            torch.autograd.set_detect_anomaly(True)
            torch.set_printoptions(precision=5, sci_mode=False)

    def train(self, xxyy, epochs):
        for e in range(epochs):
            # train predictor
            p_losses = []
            if self.predictor.needs_training():
                for xx,yy in tqdm.tqdm(xxyy):
                    y_pred = self.predictor(xx)

                    if config.EXPERIMENT_TYPE == "toptagging":
                        criterion = nn.CrossEntropyLoss(reduction='mean')
                        p_loss = criterion(y_pred, yy)
                    else:
                        p_loss = self.predictor.loss(y_pred, yy)

                    p_losses.append(float(p_loss.detach().cpu()))

                    self.predictor.optimizer.zero_grad()
                    p_loss.backward()
                    self.predictor.optimizer.step()
            p_losses = np.mean(p_losses) if len(p_losses) else 0
                
            # train basis
            b_losses = []
            b_reg = []
            """
            for xx,yy in tqdm.tqdm(xxyy):
                xp = self.basis.apply(xx)
                model_prediction = self.predictor.run(xp)

                if config.EXPERIMENT_TYPE == "toptagging":
                    criterion = nn.CrossEntropyLoss(reduction='mean')
                    b_loss = criterion(model_prediction, yy)
                else:
                    b_loss = self.basis.loss(model_prediction, yy) * config.INVARIANCE_LOSS_COEFF
                    
                # don't include regularization in outputs
                b_losses.append(float(b_loss.detach().cpu()))

                if config.EXPERIMENT_TYPE == "toptagging":
                    b_loss += torch.abs(nn.CosineSimilarity(dim=2)(xp, xx).mean())
                else:
                    b_loss += self.basis.regularization()
                
                b_reg.append(float(b_loss.detach().cpu()))

                self.basis.optimizer.zero_grad()
                b_loss.backward()
                self.basis.optimizer.step()
            """
            b_losses = np.mean(b_losses) if len(b_losses) else 0
            b_reg = np.mean(b_reg ) if len(b_reg ) else 0
        
            print("Discrete GL(n) \n", self.basis.discrete.data) 
            print("Continuous GL(n) \n", self.basis.normalized_continuous().data)
            print("Epoch", e, "Predictor loss", p_losses, "Basis loss", b_losses, "Basis reg", b_reg)

