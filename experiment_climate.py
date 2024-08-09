import torch
import torch.nn as nn
from utils import get_device
from local_symmetry import Predictor, LocalTrainer
from group_basis import GroupBasis, Homomorphism
from ff_transformer import S1FFTransformer
from config import Config
from climatenet.utils.data import ClimateDatasetLabeled, ClimateDataset, ClimateNeighborDataset
from climatenet.models import CGNet
from climatenet.utils.utils import Config
from climatenet.track_events import track_events
from climatenet.analyze_events import analyze_events
from climatenet.visualize_events import visualize_events
from os import path
from torch.utils.data import DataLoader

device = get_device()

if __name__ == '__main__':
    #config = Config()
    config = Config('config.json')

    train_path = './data/climate'
    train = ClimateNeighborDataset(path.join(train_path, 'train'), config, 5, 3)
    #loader = DataLoader(train, batch_size=64, collate_fn=ClimateNeighborDataset.collate, num_workers=4)

    # train_path = './data/climate'
    # inference_path = './data/climate'

    # train = ClimateDatasetLabeled(path.join(train_path, 'train'), config)
    # test = ClimateDatasetLabeled(path.join(train_path, 'test'), config)
    # #inference = ClimateDataset(inference_path, config)

    # cgnet = CGNet(config)
    # cgnet.train(train)
    # cgnet.evaluate(test)

    # predictor = HeatPredictor()
    # transformer = S1FFTransformer(LINE_LEN, LINE_KEY)
    # homomorphism = HeatHomomorphism()
    # basis = GroupBasis(2, transformer, homomorphism, 1, config.standard_basis, lr=5e-4)
    # dataset = HeatDataset(config.N)

    # gdn = LocalTrainer(predictor, basis, dataset, config)   
    # gdn.train()

