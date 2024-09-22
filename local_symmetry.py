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

    # implement if coset discovery is needed
    # same as `loss`, but does not collapse on first dimension
    def batched_loss(self, y_pred, y_true):
        raise NotImplemented()

    def returns_logits(self):
        return False

    # some predictors can be given as fixed functions
    def needs_training(self):
        return True


class LocalTrainer:
    def __init__(self, ff, predictor, basis, dataset, config: Config):
        self.ff = ff
        self.predictor = predictor
        self.basis = basis
        self.dataset = dataset
        self.config = config
       
        if config.debug:
            torch.autograd.set_detect_anomaly(True)
        torch.set_printoptions(precision=9, sci_mode=False)

    def loader(self):
        collate_fn = self.dataset.collate if hasattr(self.dataset, 'collate') else None
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.config.batch_size, collate_fn=collate_fn, shuffle=True)
        return loader

    def train_predictor(self, loader):
        p_losses = []
        if self.predictor.needs_training() and not self.config.reuse_predictor:
            for xx, yy in tqdm.tqdm(loader):
                xff = self.ff(xx)
                yff = self.ff(yy)

                y_pred = self.predictor.run(xff.regions(self.basis.in_rad))
                # relying on basis for radius is ugly ...
                # in rad since clipping is only needed for group basis training
                y_true = yff.regions(self.basis.in_rad)

                p_loss = self.predictor.loss(y_pred, y_true)
                p_losses.append(float(p_loss.detach().cpu()))

                self.predictor.optimizer.zero_grad()
                p_loss.backward()
                self.predictor.optimizer.step()

        p_losses = np.mean(p_losses) if len(p_losses) else 0

        if self.predictor.needs_training() and not self.config.reuse_predictor:
            torch.save(self.predictor, "predictors/" + self.predictor.name() + '.pt')

        return p_losses

    def train(self):
        loader = self.loader()

        for e in range(self.config.epochs):
            # train predictor
            p_losses = self.train_predictor(loader)

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

    def discover_cosets(self, relates, q):
        loader = self.loader()

        for e in range(self.config.epochs):
            # train predictor
            p_losses = self.train_predictor(loader)

            # train cosets
            full_losses = []
            b_losses = []
            for xx, yy in tqdm.tqdm(loader):
                xff = self.ff(xx)
                yff = self.ff(yy)

                b_loss_full = self.predictor.batched_loss(*self.basis.coset_step(xff, self.predictor))
                full_losses.append(b_loss_full.cpu().detach().numpy())
                b_loss = b_loss_full.mean()
                b_losses.append(float(b_loss))

                self.basis.optimizer.zero_grad()
                b_loss.backward()
                self.basis.optimizer.step()
            b_losses = np.mean(b_losses) if len(b_losses) else 0

            full_losses_avg = np.mean(full_losses, axis=0)
            best = np.argmin(full_losses_avg)

            print("Epoch", e, "Predictor loss", p_losses, "Best loss", full_losses_avg[best], "Best", self.basis.cosets[best].cpu().detach())
   
            if e == self.config.epochs - 1:
                inds = np.argsort(full_losses_avg)

                final = []
                for coset in self.basis.cosets[inds][:q]:
                    for curr in final:
                        if relates(curr, coset):
                            break
                    else:
                        final.append(coset)

        print("Final coset representatives", final)
        print("All", self.basis.cosets[inds][:q])
