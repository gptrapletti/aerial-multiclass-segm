import torch
from abc import ABC, abstractmethod

class BaseLoss(ABC, torch.nn.Module):
    '''Base class loss.'''
    def __init__(self, class_weights=None):
        super().__init__()
        self.activation = torch.nn.Softmax(dim=1)
        self.class_weights = class_weights
        
    def forward(self, input, target):
        input_activated = self.activation(input) # [B, C, H, W]
        input_activated = torch.where(input_activated == 0., torch.tensor(1e-8), input_activated) # to safely compute logs
        pixel_losses = self.compute_loss(input=input_activated, target=target) # [B, H, W]
        
        if self.class_weights is not None:
            weighted_pixel_losses = self.compute_weights(gt=target, class_weights=self.class_weights)
            pixel_losses = pixel_losses * weighted_pixel_losses # weight every pixel loss.
        
        # It's best practice to first compute the loss for each image separately (as mean of the 
        # losses of its pixels), and then compute the batch loss as mean of the image losses. This
        # gives each image the same importance in the training process.
        image_losses = torch.mean(pixel_losses, dim=(1, 2)) # [B]  
        batch_loss = torch.mean(image_losses) # []
        
        return batch_loss
    
    @abstractmethod
    def compute_loss(self):
        pass
    
    def compute_weights(self, target):
        '''Create weight tensor for tensor multiplication.'''
        class_weights = torch.tensor(self.class_weights)
        reshaped_class_weights = class_weights.reshape(1, class_weights.shape[0], 1, 1) # [1, C, 1, 1]
        weights = target * reshaped_class_weights # [B, C, H, W], weights only where GT=1, otherwise 0
        weights = torch.sum(weights, dim=1) # [B, H, W], sum across classes to turn to 2D
        return weights


class CrossEntropyLoss(BaseLoss):
    '''Computes the weighted cross-entropy loss between the input and target. TODO: remove args and 
    return in doc since it's a class.

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

 
class DiceLoss(BaseLoss):
    def __init__(self, class_weights=None):
        super().__init__(class_weights=class_weights)
        self.epsilon = 1e-8
        
    def forward(self, input, target):
        input_activated = self.activation(input) # [B, C, H, W]
        loss = self.compute_loss(input=input_activated, target=target) # []
        return loss
        
    def compute_loss(self, input, target):
        intersection = torch.sum(target * input, dim=[2, 3]) + self.epsilon # [B, C]
        union = torch.sum(target + input, dim=[2, 3]) + self.epsilon # [B, C]
        dice_per_image_and_class = 2 * intersection / union # [B, C]
        weights = self.compute_weights(target)               
        weighted_dice_per_image = torch.sum(dice_per_image_and_class * weights, axis=1) / torch.sum(weights, axis=1) # [B]
        batch_loss = 1 - weighted_dice_per_image.mean() # [], "1 -" to go from metric to loss
        return batch_loss
    
    def compute_weights(self, target):
        '''Create weight tensor to deal with GT channels with no objects (if a GT channel has no 
        objects, the channel weight is 0, otherwise 1). Also add class weights, if present.'''
        weights = (torch.sum(target != 0, axis=[2, 3]) != 0).type(torch.float32) # [B, C]
        if self.class_weights:
            weights = weights * torch.tensor(self.class_weights) # [B, C]        
        return weights        
        

class AerialLoss(torch.nn.Module):
    def __init__(self, loss_fn1, loss_fn2, loss_weights):
        super().__init__()
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.loss_weights = loss_weights
        self.weights = torch.tensor(self.loss_weights).to('cuda:0')
        
    def forward(self, input, target):
        loss1 = self.loss_fn1(input, target)
        loss2 = self.loss_fn2(input, target)
        losses = torch.cat([loss1.unsqueeze(0), loss2.unsqueeze(0)])
        loss = torch.sum(losses * self.weights)
        return loss       
        

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
        

