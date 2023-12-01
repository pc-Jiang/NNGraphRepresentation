from collections import OrderedDict
from distances import comparisons


def reorgnize_activation(activation_list, model_names):
    new_act_dict = {}
    for i, activation in enumerate(activation_list):
        name = model_names[i]

        for k, v in activation:
            # k is the layer to be recorded
            new_key = name + k # new_key: model+pret+layer
            new_act_dict[new_key] = v

    return new_act_dict


def get_distance(dis_name, act_dict):
    pairs = {'{}+{}'.format(k1, k2): [v1, v2] for k1, v1 in act_dict.items() for k2, v2 in act_dict.items() if k1 < k2}
    num_models = len(act_dict)
    
    if dis_name == 'LES':
        distance = comparisons.CompareLES(num_models)
    elif dis_name == 'gw':
        distance = comparisons.CompareGW(num_models)
    else:
        raise NotImplementedError(f'{dis_name} not implemented! ')
    
    for pair, data in pairs.items():
        key1, key2 = pair.split('+')
        index1, index2 = [list(act_dict.keys()).index(k) for k in (key1, key2)]
        distance.compare_dataset(data[0], data[1], index1, index2)

    distance.all_distances = distance.all_distances + distance.all_distances.T
    
    return distance.all_distances


def get_all_distances(dis_name_list, activation_list,  model_names):
    r'''
    
    :Output:
         all_dist - dict, key: distance name in dis_name_list; value: distance matrix in the order of act_dict.keys()
        row_names - act_dict.keys() indicating the columns and rows in the distance matrix
    '''
    act_dict = reorgnize_activation(activation_list,  model_names)
    act_dict = OrderedDict(act_dict)
    all_dist = {}

    for dis in dis_name_list:
        all_dist[dis] = get_distance(dis, act_dict)
    
    row_names = list(act_dict.keys())

    return all_dist, row_names
