import torch
from abc import ABC, abstractmethod

class BaseLoss(ABC, torch.nn.Module):
    '''Base class loss.'''
    def __init__(self, class_weights=None):
        super().__init__()
        self.activation = torch.nn.Softmax(dim=1)
        self.class_weights = class_weights
        
    def forward(self, input, target):
        input_activated = self.activation(input)
        input_activated = torch.where(input_activated == 0., torch.tensor(1e-8), input_activated) # to safely compute logs
        pixel_losses = self.compute_loss(input=input_activated, target=target) # [B, H, W]
        
        if self.class_weights is not None:
            weighted_pixel_losses = self.compute_weights(gt=target, class_weights=self.class_weights)
            pixel_losses = pixel_losses * weighted_pixel_losses # weight every pixel loss.
            
        batch_loss = torch.mean(pixel_losses) # batch loss as average over all pixel losses
        
        return batch_loss
    
    @abstractmethod
    def compute_loss(self):
        pass
    
    def compute_weights(self, gt, class_weights):
        '''Create weight tensor for tensor multiplication.'''
        weights = gt * class_weights.reshape(1, class_weights.shape[0], 1, 1) # [B, C, H, W]
        weights = torch.sum(weights, dim=1) # [B, H, W]
        return weights


class CrossEntropyLoss(BaseLoss):
    '''Computes the weighted cross-entropy loss between the input and target. TODO: remove args and return since it's a class

    Args:
        input: A tensor with model logits. Required shape [batch size, n classes, H, W]. 
        target: A tensor with GT one-hot-encoded. Required shape [batch size, n classes, H, W].
            
    Returns:
        torch.Tensor: A scalar tensor representing the computed weighted cross-entropy loss for the input batch,
        as average on all pixel losses.
    
    '''
    def __init__(self, class_weights=None):
        super().__init__(class_weights=class_weights)
           
    def compute_loss(self, input, target):
        pixel_losses = -torch.sum(target * torch.log(input), dim=1) # shape = [B, H, W]
        
        return pixel_losses
     
         
class FocalLoss(BaseLoss):
    def __init__(self, gamma, class_weights=None):
        super().__init__(class_weights = class_weights)
        self.gamma = gamma
               
    def compute_loss(self, input, target):
        pt = torch.where(target==1, input, 1 - input)
        pt_log = torch.log(torch.where(pt==0, 1e-8, pt)) # safely compute log
        pixel_losses = -torch.pow((1 - pt), self.gamma) * pt_log # [B, C, H, W]
        summed_pixel_losses = torch.sum(pixel_losses, dim=1) # [B, H, W]
        
        return summed_pixel_losses
    

if __name__ == '__main__':
    pred = torch.randn(size=(8, 6, 256, 256)).type(torch.float32)
    gt = torch.randint(low=0, high=2, size=(8, 6, 256, 256)).type(torch.float32)
    
    ce_fn = CrossEntropyLoss()
    ce_loss = ce_fn(pred, gt)
    print(ce_loss)
    
    torch_ce_fn = torch.nn.CrossEntropyLoss()
    torch_ce_loss = torch_ce_fn(input=pred, target=gt)
    print(torch_ce_loss)
    
    fl_fn = FocalLoss(gamma=2)
    fl_loss = fl_fn(pred, gt)
    print(fl_loss)
        

