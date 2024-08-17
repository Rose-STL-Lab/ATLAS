from abc import ABC, abstractmethod

class FeatureField(ABC):
    def __init__(self, data):
        self.data = data

    @abstractmethod
    def atlas(self):
        pass

    @abstractmethod
    def regions(self, radius):
        '''
            Should always be the same location and orientations,
            but the exact location and orientations are implementation defined
        '''
        pass

    def num_charts(self):
        return self.atlas(0).shape[1]
        
    def batch_size(self):
        return self.data.shape[0]

class UVFeatureField(FeatureField):
    def __init__(self, data):
        pass

class R2FeatureField(FeatureField):
    def __init__(self, data):
        super().__init__(data)
    
    def atlas(self):
        return torch.nn.functional.fold(self.data, 5, padding=2)

    def regions(self, radius):
        mid_r, mid_c = self.data.shape[-2] // 2, self.data.shape[-1] // 2
        return self.data[
            :,
            :, # pytorch prefers channel first
            mid_r - radius: mid_r + radius + 1,
            mid_c - radius: mid_c + radius + 1
        ].unsqueeze(1)
