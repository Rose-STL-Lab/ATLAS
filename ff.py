import torch
from abc import ABC, abstractmethod

class FeatureField(ABC):
    def __init__(self, data):
        self.data = data

    @abstractmethod
    def atlas(self):
        '''
            kernel size of 5, used in downstream cnn
        '''
        pass

    @staticmethod
    @abstractmethod
    def has_standard_atlas():
        ''' atlas is just pixelwise neighbors, in which case convolution is optimized
        '''
        pass

    @abstractmethod
    def regions(self, radius):
        '''
            Should always be the same location and orientations,
            but the exact location and orientations are implementation defined
            
            used for predictor
        '''
        pass

    def num_charts(self):
        return self.regions(0).shape[1]
        
    def batch_size(self):
        return self.data.shape[0]

class R2FeatureField(FeatureField):
    def __init__(self, data):
        super().__init__(data)

        w = self.data.shape[-1]
        h = self.data.shape[-2]
        mid_c = self.data.shape[-1] // 2
        locs = [(h * 0.5, w * 0.5)]

        self.locs = [(int(r), int(c)) for r, c in locs]
    
    def atlas(self):
        ret = torch.nn.functional.unfold(self.data, (5, 5), padding=2)
        ret = ret.view(-1, self.data.shape[1], 5, 5, *self.data.shape[2:])
        ret = ret.permute(0, 1, 4, 2, 5, 3).reshape(-1, self.data.shape[1], 5 * self.data.shape[2], 5 * self.data.shape[3])

        return ret

    @staticmethod
    def has_standard_atlas():
        return True

    def regions(self, radius):
        charts = [
            self.data[
                :,
                :, # pytorch prefers channel first
                r - radius: r + radius + 1,
                c - radius: c + radius + 1
            ] for r,c in self.locs
        ]
        return torch.stack(charts, dim=1)
