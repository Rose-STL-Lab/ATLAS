import numpy as np
import torch
from abc import ABC, abstractmethod

from utils import mae


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
    def __init__(self, predictor, basis, invariance_fac=3, reg_fac=0.1, debug=True):
        """
            predictor: local_symmetry.py/Predictor
                This corresponds to xi in the propoasl
            basis: lie_basis.py/GroupBasis
                Basis for the discovered symmetry group (somewhat corresponding to G in the proposal)
        """
        self.predictor = predictor
        self.basis = basis
        self.invariance_fac = invariance_fac
        self.reg_fac = reg_fac
       
        if debug:
            torch.autograd.set_detect_anomaly(True)
            torch.set_printoptions(precision=5, sci_mode=False)

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
            b_reg = []
            for xx, yy in xxyy:
                xp = self.basis.apply(xx)
                model_prediction = self.predictor.run(xp)

                b_loss = self.basis.loss(model_prediction, yy) * self.invariance_fac
                b_losses.append(float(b_loss.detach().cpu()))

                b_loss += self.basis.regularization() * self.reg_fac
                b_reg.append(float(b_loss.detach().cpu()))

                self.basis.optimizer.zero_grad()
                b_loss.backward()
                self.basis.optimizer.step()
            b_losses = np.mean(b_losses) if len(b_losses) else 0
            b_reg = np.mean(b_reg ) if len(b_reg ) else 0
        
            print("Discovered Basis \n", self.basis.summary())
            print("Epoch", e, "Predictor loss", p_losses, "Basis loss", b_losses, "Basis reg", b_reg)

