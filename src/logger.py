import pytorch_lightning as pl

def get_logger(experiment_name, run_name):
    '''Returns a logger.

    Args:
        experiment_name: name of the experiment
        run_name: name of the run
    '''
    logger = pl.loggers.MLFlowLogger(experiment_name=experiment_name, run_name=run_name)
    return logger
    