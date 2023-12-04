import os
import os.path as osp

from utils.exp_utils import prepare_experiments, save_json
from utils.dis_utils import get_all_distances
from configs.configs_global import RESULT_DIR
from plots import heatmap_plot
from configs.config import ExpConfig


os.makedirs(RESULT_DIR, exist_ok=True)


#%% Experiments
def compare_representation_across_models():
    # answer question: does representation for trained models related to structure?
    # assumption: ResNet18 and ResNet50 has smaller distance, AlexNet and VGG has smaller distance
    config = ExpConfig()
    config.exp_name = 'compare_across_models'
    config.model_names = ['AlexNet', 'VGG', 'ResNet18', 'ResNet50']
    config.distance_measure = ['les', 'gw', 'imd', 'stochastic']
    os.makedirs(osp.join(RESULT_DIR, config.exp_name), exist_ok=True)
    save_json(osp.join(RESULT_DIR, config.exp_name, 'config.json'), config)

    activation_dict = prepare_experiments(config)
    all_dist, row_names, running_times = get_all_distances(config.distance_measure, activation_dict)
    heatmap_plot(config.exp_name, all_dist, row_names)

    save_json(osp.join(RESULT_DIR, config.exp_name, 'distances.json'), all_dist, row_names, running_times)


def compare_learnt_unlearnt():
    # answer question: does representation for trained/untrained models related to structure?
    # assumption: trained networks has a smaller distance compared to untrained networks with same structure
    config = ExpConfig()
    config.exp_name = 'compare_learnt_unlearnt'
    config.pret = [True, False]
    config.model_names = ['AlexNet', 'VGG', 'ResNet18', 'ResNet50']
    config.distance_measure = ['les', 'gw', 'imd']
    os.makedirs(osp.join(RESULT_DIR, config.exp_name), exist_ok=True)
    save_json(osp.join(RESULT_DIR, config.exp_name, 'config.json'), config)

    activation_dict = prepare_experiments(config)
    all_dist, row_names, running_times = get_all_distances(config.distance_measure, activation_dict)
    heatmap_plot(config.exp_name, all_dist, row_names)

    save_json(osp.join(RESULT_DIR, config.exp_name, 'distances.json'), all_dist, row_names, running_times)


def compare_layers_within_model():
    # answer question: does representation for trained models vary a lot during the processing? 
    # assumption: closer layers will have smaller distances
    config = ExpConfig()
    config.exp_name = 'compare_layers'
    config.model_names = ['ResNet18', 'ResNet50']
    config.record_layers = [4, 5, 6, 7]
    config.record_input = True
    config.distance_measure = ['les', 'gw', 'imd']
    os.makedirs(osp.join(RESULT_DIR, config.exp_name), exist_ok=True)
    save_json(osp.join(RESULT_DIR, config.exp_name, 'config.json'), config)

    activation_dict = prepare_experiments(config)
    all_dist, row_names, running_times = get_all_distances(config.distance_measure, activation_dict)
    heatmap_plot(config.exp_name, all_dist, row_names)

    save_json(osp.join(RESULT_DIR, config.exp_name, 'distances.json'), all_dist, row_names, running_times)


def compare_across_classes():
    # answer question: does representation with different labels for trained models similar? 
    # assumption: representation for same labels will be smaller than cross labels
    config = ExpConfig()
    config.exp_name = 'compare_classes'
    config.record_input = True
    config.model_names = ['ResNet18', ]
    config.distance_measure = ['les', 'gw', 'imd']
    config.target_class = [0, 1]
    os.makedirs(osp.join(RESULT_DIR, config.exp_name), exist_ok=True)
    save_json(osp.join(RESULT_DIR, config.exp_name, 'config.json'), config)

    activation_dict = {}
    for target_class in config.target_class:
        act_dict = prepare_experiments(config, target_class)
        for k, v in act_dict.items():
            activation_dict[k+'_'+str(target_class)] = v
    
    all_dist, row_names, running_times = get_all_distances(config.distance_measure, activation_dict)
    heatmap_plot(config.exp_name, all_dist, row_names)
    save_json(osp.join(RESULT_DIR, config.exp_name, 'distances.json'), all_dist, row_names, running_times)


def compare_across_classes_VGG():
    # answer question: does representation with different labels for trained models similar? 
    # assumption: representation for same labels will be smaller than cross labels
    config = ExpConfig()
    config.exp_name = 'compare_classes_VGG'
    config.record_input = True
    config.model_names = ['VGG', ]
    config.distance_measure = ['les', 'gw', 'imd']
    config.target_class = [0, 1]
    os.makedirs(osp.join(RESULT_DIR, config.exp_name), exist_ok=True)
    save_json(osp.join(RESULT_DIR, config.exp_name, 'config.json'), config)

    activation_dict = {}
    for target_class in config.target_class:
        act_dict = prepare_experiments(config, target_class)
        for k, v in act_dict.items():
            activation_dict[k+'_'+str(target_class)] = v
    
    all_dist, row_names, running_times = get_all_distances(config.distance_measure, activation_dict)
    heatmap_plot(config.exp_name, all_dist, row_names)
    save_json(osp.join(RESULT_DIR, config.exp_name, 'distances.json'), all_dist, row_names, running_times)


if __name__ == '__main__':
    compare_representation_across_models()

# %%
