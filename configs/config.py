class ExpConfig(object):
    def __init__(self) -> None:
        self.num_wkr = 2
        self.batch_size = 64
        self.dataset = 'CIFAR10'
        self.target_class = [0, ]
        self.num_samples = 500
        self.exp_name = ''
        # self.iter_num = 3

        self.pret = [True, ]
        self.model_names = ['AlexNet', ]
        self.record_layers = [-3, ]
        self.distance_measure = ['les', ]
        self.record_input = False
        