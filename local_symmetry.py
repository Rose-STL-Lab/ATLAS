import numpy as np
import torch
import tqdm
from abc import ABC, abstractmethod

from config import Config
from utils import rmse


class Predictor(ABC):
    optimizer = None

    # alias for run
    def __call__(self, x):
        return self.run(x)

    @abstractmethod
    def run(self, x):
        pass

    @abstractmethod
    def name(self):
        pass

    def loss(self, y_pred, y_true):
        return rmse(y_pred, y_true)

    def returns_logits(self):
        return False

    # some predictors can be given as fixed functions
    def needs_training(self):
        return True


class LocalTrainer:
    def __init__(self, ff, predictor, basis, dataset, config: Config, debug=True):
        self.ff = ff
        self.predictor = predictor
        self.basis = basis
        self.dataset = dataset
        self.config = config
       
        if debug:
            torch.autograd.set_detect_anomaly(True)
            torch.set_printoptions(precision=9, sci_mode=False)

    def train(self):
        collate_fn = self.dataset.collate if hasattr(self.dataset, 'collate') else None
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.config.batch_size, collate_fn=collate_fn, shuffle=True)

        for e in range(self.config.epochs):
            # train predictor
            p_losses = []
            if self.predictor.needs_training() and not self.config.reuse_predictor:
                for _ in range(20):
                    for xx, yy in tqdm.tqdm(loader):
                        xff = self.ff(xx)
                        yff = self.ff(yy)

                        # relying on basis for radius is ugly ...
                        y_pred = self.predictor.run(xff.regions(self.basis.in_rad))
                        y_true = yff.regions(self.basis.out_rad)

                        p_loss = self.predictor.loss(y_pred, y_true)
                        p_losses.append(float(p_loss.detach().cpu()))

                        self.predictor.optimizer.zero_grad()
                        p_loss.backward()
                        self.predictor.optimizer.step()

            p_losses = np.mean(p_losses) if len(p_losses) else 0

            if self.predictor.needs_training() and not self.config.reuse_predictor:
                torch.save(self.predictor, "predictors/" + self.predictor.name() + '.pt')

            # train basis
            b_losses = []
            b_reg = []
            for xx, yy in tqdm.tqdm(loader):
                xff = self.ff(xx)
                yff = self.ff(yy)

                b_loss = self.basis.step(xff, self.predictor, yff) 
                b_losses.append(float(b_loss))

                reg = self.basis.regularization(e)
                b_loss += reg
                b_reg.append(float(reg))

                self.basis.optimizer.zero_grad()
                b_loss.backward()
                self.basis.optimizer.step()
            b_losses = np.mean(b_losses) if len(b_losses) else 0
            b_reg = np.mean(b_reg) if len(b_reg) else 0

            print("Discovered Basis \n", self.basis.summary())
            print("Epoch", e, "Predictor loss", p_losses, "Basis loss", b_losses, "Basis reg", b_reg)

