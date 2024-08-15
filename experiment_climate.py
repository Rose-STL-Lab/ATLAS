import torch
import torch.nn as nn
from utils import get_device
from local_symmetry import Predictor, LocalTrainer
from group_basis import GroupBasis, TrivialHomomorphism
from ff_transformer import SingletonFFTransformer
from config import Config
from climatenet.utils.data import ClimateDatasetLabeled, ClimateDataset, ClimateNeighborDataset
from climatenet.models import CGNet, CGNetModule
from climatenet.utils.losses import jaccard_loss
from climatenet.utils.metrics import get_cm, get_iou_perClass
from os import path
from torch.utils.data import DataLoader
from torch.optim import Adam
import tqdm
import numpy as np

device = get_device()

class ClimatePredictor(Predictor):
    def __init__(self):
        super().__init__()
        self.network = CGNetModule(classes=config.label_length, channels=config.field_length).cuda()
        self.optimizer = Adam(self.network.parameters(), lr=config.lr)   
    
    def __call__(self, x):
        return self.run(x)

    def run(self, x):
        return self.network(x)
    
    def loss(self, y_pred, y_true):
        return jaccard_loss(y_pred, y_true)

    def needs_training(self):
        return True

def train():
    loader = DataLoader(train_data, batch_size=config.batch_size, collate_fn=ClimateNeighborDataset.collate, num_workers=4, shuffle=True)

    for e in range(config.epochs):

        # train predictor
        aggregate_cms = []
        for _ in range(num_charts):
            aggregate_cms.append(np.zeros((3,3)))

        for features, labels in tqdm.tqdm(loader):
            features = torch.tensor(features.values).float().cuda()
            labels = torch.tensor(labels.values).float().cuda()

            # Train each pointwise predictors
            for i in range(num_charts):
                predictor = predictors[i]
                predictor.network.train()

                # Push data on GPU and pass forward
                sliced_feature = features[:, :, i, :, :].cuda()
                sliced_labels = labels[:, i, :, :].cuda()
                outputs = torch.softmax(predictor.network(sliced_feature), 1)
                
                # Update training CM
                predictions = torch.max(outputs, 1)[1]
                aggregate_cms[i] += get_cm(predictions, labels, 3)
                
                # Pass backward
                p_loss = predictor.loss(outputs, sliced_labels)
                p_loss.backward()
                predictor.optimizer.step()
                predictor.optimizer.zero_grad() 

        for i in range(num_charts):
            #print("Chart Number:", i+1)
            #print(aggregate_cms[i])
            ious = get_iou_perClass(aggregate_cms[i])
            print('IOUs: ', ious, ', mean: ', ious.mean())

        # train basis
        b_losses = []
        b_reg = []
        # for xx, yy in tqdm.tqdm(loader):
        #     xp, yp = basis.apply(xx, yy)
        #     model_prediction = predictor.run(xp)

        #     b_loss = basis.loss(model_prediction, yp) 
        #     b_losses.append(float(b_loss.detach().cpu()))

        #     reg = basis.regularization(e)
        #     b_loss += reg
        #     b_reg.append(float(reg))

        #     basis.optimizer.zero_grad()
        #     b_loss.backward()
        #     basis.optimizer.step()
        b_losses = np.mean(b_losses) if len(b_losses) else 0
        b_reg = np.mean(b_reg) if len(b_reg) else 0

        #print("Discovered Basis \n", basis.summary())
        #print("Epoch", e, "Predictor loss", p_losses, "Basis loss", b_losses, "Basis reg", b_reg)


if __name__ == '__main__':
    config = Config()
    
    # Parameters
    num_charts = 3
    chart_size = 128
    train_path = './data/climate'

    train_data = ClimateNeighborDataset(path.join(train_path, 'train'), config.fields, chart_size, num_charts)

    predictors = []
    for _ in range(num_charts):
        predictors.append(ClimatePredictor())

    # transformer = SingletonFFTransformer((config.n_component, ))
    # homomorphism = TrivialHomomorphism([1], 1)
    # basis = GroupBasis(4, transformer, homomorphism, 7, config.standard_basis, loss_type='cross_entropy')

    train()

