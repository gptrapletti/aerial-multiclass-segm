# Use `mlflow gc` before.

import os
import shutil

mlflow_log_path = 'logs/mlruns'
hydra_log_path = 'logs/hydra/aerial-multiclass-segm/'

discard = ['models', '0', '.trash']

experiment_id = [x for x in os.listdir(mlflow_log_path) if x not in discard][0]

run_ids = os.listdir(os.path.join(mlflow_log_path, experiment_id))

run_names = []
for run_id in run_ids:
    run_id_dirpath = os.path.join(mlflow_log_path, experiment_id, run_id)
    if os.path.isdir(run_id_dirpath):
        run_name_filepath = os.path.join(run_id_dirpath, 'tags', 'mlflow.runName')
        with open(run_name_filepath, 'r') as f:
            run_name = f.read()
            run_names.append(run_name)
            
for run_name in os.listdir(hydra_log_path):
    if run_name not in run_names:
        shutil.rmtree(os.path.join(hydra_log_path, run_name))