import hydra

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