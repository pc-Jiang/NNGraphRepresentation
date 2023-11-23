import torch
import torchvision
from configs_global import DEVICE


class ActivationHook:
    def __init__(self, layer) -> None:

        self.activation = []
        self.handle = layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, inp, outp):
        op = outp.clone().detach().cpu().numpy()
        self.activation.append(op.reshape(1, -1))
        # self.activation.append(op)


def get_named_layers(model, record_layers):

    handles = {}
    all_layers = list(model.named_children())
    for id in record_layers:
        name, layer = all_layers[id]
        handles[name] = ActivationHook(layer)
    
    return handles


def get_pret_models(model_name, pret=True, record_layers=[-2,]):
    r"""
    :Input:
           model_name - name of model
                 pret - True if load the pretrained weights, False if load random initialized weights
        record_layers - which layer to record. In the models implemented in this function, 
                        -1 is the classifier layer, -2 is the average pooling layer, and -3 is the last feature layer in the network. 
                        For AlexNet and VGG, only one feature layer (0 or -3)
                        For ResNet, layer 4-7 are the Residual blocks, -3 is the 7th layer
    
    :Output: 
            net - model defined by model_name
        weights - weights of net, should run `weights.transforms()` as a preprocessing tool for data
        handles - handles for the pytorch hook to record network activity, should run `remove(handles($layer_name))` after recording the activations
    """

    if model_name == 'ResNet18':
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        net = torchvision.models.resnet18(weights=(weights if pret else None)).to(DEVICE)
        
    elif model_name == 'ResNet50':
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        net = torchvision.models.resnet50(weights=(weights if pret else None)).to(DEVICE)
        
    elif model_name == 'AlexNet':
        weights = torchvision.models.AlexNet_Weights.DEFAULT
        net = torchvision.models.alexnet(weights=(weights if pret else None)).to(DEVICE)
        
    elif model_name == 'VGG':
        weights = torchvision.models.VGG11_Weights.DEFAULT
        net = torchvision.models.vgg11(weights=(weights if pret else None)).to(DEVICE)
       
    else:
        raise NotImplementedError(f"{model_name} model not implemented!")
    
    handles = get_named_layers(net, record_layers)
    net.eval()

    # preprocess = weights.transforms()
    # img = preprocess(img)
    return net, weights.transforms(), handles
