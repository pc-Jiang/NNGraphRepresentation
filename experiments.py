import os
import os.path as osp

from utils.exp_utils import prepare_experiments, save_json, load_json
from utils.dis_utils import get_all_distances
from configs.configs_global import RESULT_DIR
from plots import heatmap_plot
from configs.config import ExpConfig


os.makedirs(RESULT_DIR, exist_ok=True)


def run_experiment(config):
    if len(config.target_class) >1 :
        activation_dict = {}
        for target_class in config.target_class:
            act_dict = prepare_experiments(config, target_class)
            for k, v in act_dict.items():
                activation_dict[k+'_'+str(target_class)] = v
    else:
        activation_dict = prepare_experiments(config)

    all_dist, row_names, running_times = get_all_distances(config.distance_measure, activation_dict)
    heatmap_plot(config.exp_name, all_dist, row_names)

    os.makedirs(config.save_path, exist_ok=True)
    save_json(osp.join(config.save_path, 'config.json'), config)    
    save_json(osp.join(config.save_path, 'distances.json'), all_dist, row_names, running_times)


#%% Experiments
def compare_representation_across_models():
    # answer question: does representation for trained models related to structure?
    # assumption: ResNet18 and ResNet50 has smaller distance, AlexNet and VGG has smaller distance
    config = ExpConfig()
    config.exp_name = 'compare_across_models'
    config.model_names = ['AlexNet', 'VGG', 'ResNet18', 'ResNet50']
    config.distance_measure = ['les', 'imd', ]
    config.save_path = osp.join(RESULT_DIR, config.exp_name)
    return config


def compare_representation_across_models_gw():
    # answer question: does representation for trained models related to structure?
    # assumption: ResNet18 and ResNet50 has smaller distance, AlexNet and VGG has smaller distance
    config = ExpConfig()
    config.exp_name = 'compare_across_models_gw'
    config.model_names = ['AlexNet', 'VGG', 'ResNet18', 'ResNet50']
    config.distance_measure = ['gw', ]
    config.save_path = osp.join(RESULT_DIR, config.exp_name)
    return config


def compare_learnt_unlearnt():
    # answer question: does representation for trained/untrained models related to structure?
    # assumption: trained networks has a smaller distance compared to untrained networks with same structure
    config = ExpConfig()
    config.exp_name = 'compare_learnt_unlearnt'
    config.pret = [True, False]
    config.model_names = ['AlexNet', 'VGG', 'ResNet18', 'ResNet50']
    config.distance_measure = ['les', 'imd']
    config.save_path = osp.join(RESULT_DIR, config.exp_name)
    return config


def compare_learnt_unlearnt_gw():
    # answer question: does representation for trained/untrained models related to structure?
    # assumption: trained networks has a smaller distance compared to untrained networks with same structure
    config = ExpConfig()
    config.exp_name = 'compare_learnt_unlearnt_gw'
    config.pret = [True, False]
    config.model_names = ['AlexNet', 'VGG', 'ResNet18', 'ResNet50']
    config.distance_measure = ['gw', ]
    config.save_path = osp.join(RESULT_DIR, config.exp_name)
    return config


def compare_layers_within_model():
    # answer question: does representation for trained models vary a lot during the processing? 
    # assumption: closer layers will have smaller distances
    config = ExpConfig()
    config.exp_name = 'compare_layers'
    config.model_names = ['ResNet18', 'ResNet50']
    config.record_layers = [4, 5, 6, 7]
    config.record_input = True
    config.distance_measure = ['les', 'imd', ]
    config.save_path = osp.join(RESULT_DIR, config.exp_name)
    return config


def compare_layers_within_model_gw():
    # answer question: does representation for trained models vary a lot during the processing? 
    # assumption: closer layers will have smaller distances
    config = ExpConfig()
    config.exp_name = 'compare_layers_gw'
    config.model_names = ['ResNet18', 'ResNet50']
    config.record_layers = [4, 5, 6, 7]
    config.record_input = False
    config.distance_measure = ['gw', ]
    config.save_path = osp.join(RESULT_DIR, config.exp_name)
    return config


def compare_across_classes():
    # answer question: does representation with different labels for trained models similar? 
    # assumption: representation for same labels will be smaller than cross labels
    config = ExpConfig()
    config.exp_name = 'compare_classes'
    config.record_input = True
    config.model_names = ['ResNet18', 'VGG']
    config.distance_measure = ['les', 'imd', ]
    config.target_class = [0, 1]
    config.save_path = osp.join(RESULT_DIR, config.exp_name)
    return config


def compare_across_classes_gw():
    # answer question: does representation with different labels for trained models similar? 
    # assumption: representation for same labels will be smaller than cross labels
    config = ExpConfig()
    config.exp_name = 'compare_classes_gw'
    config.record_input = True
    config.model_names = ['ResNet18', 'VGG']
    config.distance_measure = ['gw', ]
    config.target_class = [0, 1]
    config.save_path = osp.join(RESULT_DIR, config.exp_name)
    return config


def get_config_list():
    config_list = [
        compare_learnt_unlearnt(),
        # compare_learnt_unlearnt_gw(),
        compare_layers_within_model(),
        # compare_layers_within_model_gw(),
        compare_across_classes(),
        # compare_across_classes_gw(),
    ]

    return config_list


def replot():
    config_list = get_config_list()
    for config in config_list:
        try:
            results = load_json(osp.join(config.save_path, 'distances.json'))
            heatmap_plot(config.exp_name, results[0], results[1])
        except:
            run_experiment(config)


if __name__ == '__main__':
    compare_representation_across_models()
