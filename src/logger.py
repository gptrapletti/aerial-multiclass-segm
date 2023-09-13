import pytorch_lightning as pl

# def get_logger(experiment_name, run_name):
#     '''Returns a logger.

#     Args:
#         experiment_name: name of the experiment
#         run_name: name of the run
#     '''
#     logger = pl.loggers.MLFlowLogger(experiment_name=experiment_name, run_name=run_name)
#     return logger

def get_logger(experiment_name, run_id):
    '''Returns a logger.

    Args:
        experiment_name: name of the experiment
        run_id: name of the run
    '''
    logger = pl.loggers.MLFlowLogger(experiment_name=experiment_name, run_id=run_id)
    return logger
    
if __name__ == '__main__':
    
    import yaml
    
    with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)

    logger = get_logger(experiment_name=cfg['experiment_name'], run_name=cfg['run_name'])
    print(logger.experiment_id, logger.run_id)