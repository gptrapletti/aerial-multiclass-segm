import hydra
from numpy import log
from pytorch_lightning.callbacks import Callback

def instantiate_callbacks(callbacks_cfg):
    """
    Takes cfg's callbacks configuration dictionary, where
    keys are callbacks name and values are their configs, and
    instantiates every callback, appending it to a list.

    Args:
        callbacks_cfg: Hydra callbacks config dictionary.

    Returns:
        List of callbacks.
    """
    callbacks = []

    for callback_name in callbacks_cfg: 
        callbacks.append(hydra.utils.instantiate(callbacks_cfg[callback_name]))

    return callbacks


class LogGradNormCallback(Callback):
    """
    Logs the gradient log norm.
    Source: https://github.com/Lightning-AI/pytorch-lightning/issues/1462
    """

    def on_after_backward(self, trainer, model):
        model.log("grad_norm", self.log_gradient_norm(model))

    def log_gradient_norm(self, model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                # Compute norm (a scalar) of parameter tensor
                param_norm = p.grad.detach().data.norm(2)
                # Square it according to L2 norm formula, then add
                total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5
        log_grad_norm = log(total_norm + 1e-6)
        return log_grad_norm