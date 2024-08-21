import torch
from abc import ABC, abstractmethod

class FeatureField(ABC):
    def __init__(self, data):
        self.data = data

    @abstractmethod
    def atlas(self):
        '''
            kernel size of 5
        '''
        pass

    @abstractmethod
    def regions(self, radius):
        '''
            Should always be the same location and orientations,
            but the exact location and orientations are implementation defined
        '''
        pass

    def num_charts(self):
        return self.regions(0).shape[1]
        
    def batch_size(self):
        return self.data.shape[0]

class UVFeatureField(FeatureField):
    def __init__(self, data):
        pass

class R2FeatureField(FeatureField):
    def __init__(self, data):
        super().__init__(data)
    
    def atlas(self):
        ret = torch.nn.functional.unfold(self.data, (5, 5), padding=2)
        ret = ret.view(-1, self.data.shape[1], 5, 5, *self.data.shape[2:])
        ret = ret.permute(0, 1, 4, 2, 5, 3).reshape(-1, self.data.shape[1], 5 * self.data.shape[2], 5 * self.data.shape[3])

        return ret

    def regions(self, radius):
        w = self.data.shape[-1]
        h = self.data.shape[-2]
        mid_c = self.data.shape[-1] // 2
        locs = [(h * 0.375, w * 0.375), (0.625 * h, 0.625 * w), (h // 4, 3 * w // 4), (3 * h // 4, 3 * w // 4), (h // 2, w // 2)][:2]
        locs = [(int(r), int(c)) for r, c in locs]
        charts = [
            self.data[
                :,
                :, # pytorch prefers channel first
                r - radius: r + radius + 1,
                c - radius: c + radius + 1
            ] for r,c in locs
        ]
        return torch.stack(charts, dim=1)
