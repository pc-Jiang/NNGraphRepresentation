from collections import OrderedDict
from distances import comparisons
import time


def get_distance(dis_name, act_dict):
    pairs = {'{}+{}'.format(k1, k2): [v1, v2] for k1, v1 in act_dict.items() for k2, v2 in act_dict.items() if k1 < k2}
    num_models = len(act_dict)
    
    if dis_name == 'les':
        distance = comparisons.CompareLES(num_models)
    elif dis_name == 'gw':
        distance = comparisons.CompareGW(num_models)
    elif dis_name == 'imd':
        distance = comparisons.CompareIMD(num_models)
    elif dis_name == 'stochastic':
        pass
    else:
        raise NotImplementedError(f'{dis_name} not implemented! ')
    
    for pair, data in pairs.items():
        key1, key2 = pair.split('+')
        index1, index2 = [list(act_dict.keys()).index(k) for k in (key1, key2)]
        distance.compare_dataset(data[0], data[1], index1, index2)

    distance.all_distances = distance.all_distances + distance.all_distances.T
    
    return distance.all_distances


def get_all_distances(dis_name_list, act_dict):
    r'''
    
    :Output:
         all_dist - dict, key: distance name in dis_name_list; value: distance matrix in the order of act_dict.keys()
        row_names - act_dict.keys() indicating the columns and rows in the distance matrix
    '''
    print('Starting calculating distances! ')

    act_dict = OrderedDict(act_dict)
    all_dist = {}

    running_times = []
    for dis in dis_name_list:
        st_time = time.time()
        all_dist[dis] = get_distance(dis, act_dict)
        print(f'Finish calculating {dis} distances! ')
        end_time = time.time()
        running_times.append(end_time-st_time)

    row_names = list(act_dict.keys())
    print('Finish calculating distances! ')

    return all_dist, row_names, running_times
