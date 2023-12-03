import json
import numpy as np

from models import get_pret_models
from datasets import get_image_dataset_subset, model_representation
from configs.config import ExpConfig


def prepare_experiments(config):
    print('Start record model activations! ')
    model_name_list = []
    handles_list = []
    model_list = []

    for pret in config.pret:
        for name in config.model_names:
            net, transform, handles = get_pret_models(name, pret, record_layers=config.record_layers)
            model_name_list.append(name + ('T' if pret else 'F'))
            model_list.append(net)
            handles_list.append(handles)
    
    dl = get_image_dataset_subset(config.dataset, transform, config.batch_size, config.target_class, config.num_wkr)

    activation_list = model_representation(model_list, handles_list, dl, config.num_samples, config.record_input)
    print('Get all model activations! ')
    
    return model_name_list, activation_list


def save_json(file_path, *args):
    data2save = []

    for arg in args:
        if isinstance(arg, np.ndarray):
            arg = arg.tolist()
        elif isinstance(arg, dict):
            for k, v in arg.items():
                arg[k] = v.tolist()
        elif isinstance(arg, ExpConfig):
            arg = arg.__dict__
        data2save.append(arg)

    with open(file_path, 'w') as f:
        json.dump(data2save, f)
