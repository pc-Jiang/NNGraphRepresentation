import json
from exp_utils import prepare_experiments
from distances.dis_utils import get_all_distances
from configs_global import RESULT_DIR


class ExpConfig(object):
    def __init__(self) -> None:
        self.num_wkr = 4
        self.batch_size = 64
        self.dataset = 'CIFAR10'
        self.target_class = [0, ]
        self.num_samples = 1000
        # self.iter_num = 3

        self.pret = [True, ]
        self.model_names = ['AlexNet', ]
        self.record_layers = [-2, ]
        self.distance_measure = ['les', ]


def save_json():
    pass


def compare_representation_across_models():
    # answer question: does representation for trained models related to structure?
    # assumption: ResNet18 and ResNet50 has smaller distance, AlexNet and VGG has smaller distance
    config = ExpConfig()
    config.model_names = ['AlexNet', 'VGG', 'ResNet18', 'ResNet50']
    config.distance_measure = ['les', 'gw']
    model_name_list, activation_list = prepare_experiments(config)
    all_dist, row_names = get_all_distances(config.distance_measure, activation_list,  model_name_list)
    # TODO: heatmap plot for distance matrix


def compare_learnt_unlearnt():
    # answer question: does representation for trained/untrained models related to structure?
    # assumption: trained networks has a smaller distance compared to untrained networks with same structure
    config = ExpConfig()
    config.pret = [True, False]
    pass


def compare_layers_within_model():
    # answer question: does representation for trained models vary a lot during the processing? 
    # assumption: closer layers will have smaller distances
    # TODO: how to add original data into consideration? 
    config = ExpConfig()
    config.model_names = ['ResNet18', 'ResNet50']
    config.record_layers = [4, 5, 6, 7]
    pass


def compare_across_classes():
    # answer question: does representation with different labels for trained models similar? 
    # assumption: representation for same labels will be smaller than cross labels
    pass


if __name__ == '__main__':
    compare_representation_across_models()
