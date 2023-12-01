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


def compare_representation_across_models():
    config = ExpConfig()
    config.model_names = ['AlexNet', 'VGG', 'ResNet18', 'ResNet50']
    config.distance_measure = ['les', 'gw']
    model_name_list, activation_list = prepare_experiments(config)
    all_dist, row_names = get_all_distances(config.distance_measure, activation_list,  model_name_list)


def compare_learnt_unlearnt():
    config = ExpConfig()
    config.pret = [True, False]
    pass


def compare_layers_within_model():
    config = ExpConfig()
    config.model_names = ['ResNet18', 'ResNet50']
    config.record_layers = [4, 5, 6, 7]
    pass


def compare_across_classes():
    pass


if __name__ == '__main__':
    compare_representation_across_models()
