import torch

class RandomBBoxSampler(torch.utils.data.Sampler):
    '''Sampler class used to re-initialize the bbox list at
    the start of each epoch, so that different epochs have different
    patches. Without this sampler, the DataModule class would instantiate
    the Dataset object and then the bbox would be the same for all epochs. 
    '''
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __iter__(self):
        self.dataset.reset_patch_bboxs()
        return iter(range(len(self.dataset)))
    
    def __len__(self):
        return len(self.dataset)      
