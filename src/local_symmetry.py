import numpy as np
import torch
from abc import ABC, abstractmethod

from utils import mae
import config

class Basis(ABC):
    optimizer = None

    # returns (g*x, regularization)
    @abstractmethod
    def apply(self, x, y):
        pass 

    def loss(self, x, y):
        return mae(x, y)

class Predictor(ABC):
    optimizer = None

    @abstractmethod
    def run(self, x):
        pass

    def loss(self, x, y):
        return mae(x, y)

    # some predictors can be given as fixed functions
    def needs_training(self):
        return True

    def __call__(self, x):
        return self.run(x)

class LocalTrainer:
    def __init__(self, predictor, basis):
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
                    p_losses.append(p_loss.detach().numpy())
                    
                    self.predictor.optimizer.zero_grad()
                    p_loss.backward()
                    self.predictor.optimizer.step()
            p_losses = np.mean(p_losses) if len(p_losses) else 0
                
            # train basis
            b_losses = []
            for xx, yy in xxyy:
                xp, regularization = self.basis.apply(xx)
                model_prediction = self.predictor(xp)

                b_loss = self.basis.loss(model_prediction, yy) * config.INVARIANCE_LOSS_COEFF
                # don't include regularization in outputs
                b_losses.append(b_loss.detach().numpy())

                b_loss += regularization

                self.basis.optimizer.zero_grad()
                b_loss.backward()
                self.basis.optimizer.step()
            b_losses = np.mean(b_losses) if len(b_losses) else 0
        
            print("Model Prediction", model_prediction, "YY", yy)

            print("Discrete", self.basis.discrete.data, "Dets", torch.det(self.basis.discrete.data))
            print("Continuous", torch.matrix_exp(self.basis.normalized_continuous().data))
            print("Epoch", e, "Predictor loss", p_losses, "Basis loss", b_losses)

