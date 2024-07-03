import numpy as np
import torch
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

    def train(self, xxyy, epochs):
        for e in range(epochs):
            # train predictor
            p_losses = []
            if self.predictor.needs_training():
                for xx, yy in xxyy:
                    y_pred = self.predictor(xx)
                    p_loss = self.predictor.loss(y_pred, yy)
                    p_losses.append(float(p_loss.detach().cpu()))

                    self.predictor.optimizer.zero_grad()
                    p_loss.backward()
                    self.predictor.optimizer.step()
            p_losses = np.mean(p_losses) if len(p_losses) else 0
                
            # train basis
            b_losses = []
            for xx, yy in xxyy:
                xp = self.basis.apply(xx)
                model_prediction = self.predictor.run(xp)

                b_loss = self.basis.loss(model_prediction, yy) * config.INVARIANCE_LOSS_COEFF
                # don't include regularization in outputs
                b_losses.append(float(b_loss.detach().cpu()))

                b_loss += self.basis.regularization()

                self.basis.optimizer.zero_grad()
                b_loss.backward()
                self.basis.optimizer.step()
            b_losses = np.mean(b_losses) if len(b_losses) else 0
        
            print("Discrete GL(n)", self.basis.discrete.data) 
            print("Continuous GL(n)", self.basis.normalized_continuous().data)
            print("Epoch", e, "Predictor loss", p_losses, "Basis loss", b_losses)

