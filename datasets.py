import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from configs_global import DATA_DIR, DEVICE


def get_image_dataset_subset(dataset_name='CIFAR10',
                      transform=transforms.ToTensor(),
                      batch_size=16, 
                      labels=[0, ],
                      num_wks=0):
    r"""
    Get the dataset subset defined by the class label
    """
    if dataset_name == 'CIFAR10':
        # classes = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer',
        #            5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
        dataset_all = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
        collate_f = None
    
    elif dataset_name == 'ImageNet1k':
        # should run `kaggle competitions download -c imagenet-object-localization-challenge` in the command line
        pass

    else:
        raise NotImplementedError(f"{dataset_name} dataset not implemented! ")
    
    subset_idx = []
    for l in labels:
        subset_idx += [i for i, (_, c) in enumerate(dataset_all) if c == l]
    dataset = Subset(dataset_all, subset_idx)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_wks, drop_last=True, collate_fn=collate_f)
    return data_loader


def model_representation(model_list, handles_list, data_loader, max_num_data=1000):
    r"""
    A function to get the model representations 
    :Input:
               model_list - List of the neural network model to be investigated
             handles_list - List of hook functions dicts for target layers {layer_name: hook_fn}
              data_loader - data loader to load the image data
             max_num_data - number of samples that is to record activation

    :Output:
        activation - numpy array shape [n, $dim_features$] that each row is a flattened representation recorded from handles
    """
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, _ = data
            for model in model_list:
                # to make sure all model receive the same input
                # weights.transforms() for weights in torchvision are the same at least for the models we're using
                outputs = model(inputs.to(DEVICE))

            if (i+1) * data_loader.batch_size > max_num_data:
                break
    
    activation_list = []
    for handles in handles_list:
        activation = {}
        for k, v in handles.items():
            # k - layer name
            activation[k] = np.concatenate(v.activation, axis=0)
            v.handle.remove()
        activation_list.append(activation)
    
    return activation_list


if __name__ == '__main__':
    from models import get_pret_models
    model, transform, handles = get_pret_models('VGG')
    data_loader = get_image_dataset_subset('CIFAR10', transform)
    activation = model_representation([model,], [handles,], data_loader, max_num_data=20)
