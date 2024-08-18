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

        # import matplotlib.pyplot as plt
        # plt.imshow(ret.detach().cpu().permute(0, 2, 3, 1).numpy()[0])
        # plt.show()

        return ret

    def regions(self, radius):
        mid_r, mid_c = self.data.shape[-2] // 2, self.data.shape[-1] // 2
        return self.data[
            :,
            :, # pytorch prefers channel first
            mid_r - radius: mid_r + radius + 1,
            mid_c - radius: mid_c + radius + 1
        ].unsqueeze(1)
