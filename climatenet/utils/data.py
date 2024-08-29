from torch.utils.data import Dataset
from os import listdir, path
import xarray as xr
from climatenet.utils.utils import Config
import numpy as np
import random as rand

class ClimateDataset(Dataset):
    '''
    The basic Climate Dataset class. 

    Parameters
    ----------
    path : str
        The path to the directory containing the dataset (in form of .nc files)
    config : Config
        The model configuration. This allows to automatically infer the fields we are interested in 
        and their normalisation statistics

    Attributes
    ----------
    path : str
        Stores the Dataset path
    fields : dict
        Stores a dictionary mapping from variable names to normalisation statistics
    files : [str]
        Stores a sorted list of all the nc files in the Dataset
    length : int
        Stores the amount of nc files in the Dataset
    '''
  
    def __init__(self, path: str, config: Config):
        self.path: str = path
        self.fields: dict = config.fields
        
        self.files: [str] = [f for f in sorted(listdir(self.path)) if f[-3:] == ".nc"]
        self.length: int = len(self.files)
      
    def __len__(self):
        return self.length

    def normalize(self, features: xr.DataArray):
        for variable_name, stats in self.fields.items():   
            var = features.sel(variable=variable_name).values
            var -= stats['mean']
            var /= stats['std']

    def get_features(self, dataset: xr.Dataset):
        features = dataset[list(self.fields)].to_array()
        self.normalize(features)
        return features.transpose('time', 'variable', 'lat', 'lon')

    def __getitem__(self, idx: int):
        file_path: str = path.join(self.path, self.files[idx]) 
        dataset = xr.load_dataset(file_path)
        return self.get_features(dataset)

    @staticmethod
    def collate(batch):
        return xr.concat(batch, dim='time')

class ClimateDatasetLabeled(ClimateDataset):
    '''
    The labeled Climate Dataset class. 
    Corresponds to the normal Climate Dataset, but returns labels as well and batches accordingly
    '''

    def __getitem__(self, idx: int):
        file_path: str = path.join(self.path, self.files[idx]) 
        dataset = xr.load_dataset(file_path)
        return self.get_features(dataset), dataset['LABELS']

    @staticmethod 
    def collate(batch):
        data, labels = map(list, zip(*batch))
        return xr.concat(data, dim='time'), xr.concat(labels, dim='time')


## New Dataset for local symmetry
class ClimateNeighborDataset(ClimateDataset):
    '''
    Attributes
    ----------
    chart_size : int
        height, width of the chart
    num_chart : int
        Number of charts in the atlas
    '''

    def __init__(self, path: str, fields, chart_size, num_chart):
        self.path: str = path
        self.fields: dict = fields
        
        self.files: [str] = [f for f in sorted(listdir(self.path)) if f[-3:] == ".nc"]
        self.length: int = len(self.files)
        self.chart_size = chart_size
        self.num_chart = num_chart
        self.charts = self.generate_charts(self.chart_size, self.num_chart)

    def generate_charts(self, chart_size, num_chart):
        pairs = []
        for _ in range(num_chart):
            r = rand.randint(0, 768-chart_size)
            c = rand.randint(0, 1152-chart_size)
            pairs.append((r, c))
        return pairs

    def get_features(self, dataset: xr.Dataset, labels):
        features = dataset[list(self.fields)].to_array()
        self.normalize(features)
        features = features.transpose('time', 'variable', 'lat', 'lon')

        # size of (1, length of variable, number of charts, chart_size, chart_size)
        new_features = np.zeros((len(features['time']), len(features['variable']), self.num_chart, self.chart_size, self.chart_size))

        for i in range(len(features['variable'])):
            variable = features[0, i, :, :]

            for j, (r,c) in enumerate(self.charts):
                new_features[0, i, j] = variable[r:r+self.chart_size, c:c+self.chart_size].values

        time = features['time'].values
        variables = features['variable'].values
        chart = list(range(self.num_chart))
        temp_lat = temp_lon = list(range(self.chart_size))
        atlas_feature = xr.DataArray(new_features, coords=[time, variables, chart, temp_lat, temp_lon], dims=['time', 'variable', 'chart', 'lat', 'lon'])

        # size of (number of charts, chart_size, chart_size)
        new_labels = []
        for (r,c) in self.charts:
            new_labels.append(labels[r:r+self.chart_size, c:c+self.chart_size].values)
            #new_labels.append(labels.values)

        atlas_label = xr.DataArray(new_labels, coords=[chart, temp_lat, temp_lon], dims=['chart', 'lat', 'lon'])

        return atlas_feature, atlas_label

    # Return an array of charts
    def __getitem__(self, idx: int):
        file_path: str = path.join(self.path, self.files[idx]) 
        dataset = xr.load_dataset(file_path)
        return self.get_features(dataset, dataset['LABELS'])

    @staticmethod 
    def collate(batch):
        data, labels = map(list, zip(*batch))
        return xr.concat(data, dim='time'), xr.concat(labels, dim='time')