import torch
from collections import OrderedDict
from model import *
import copy


def add2model(model1, model2):
    res = OrderedDict()
    # ignore error that model1.state_dict().keys != model2.
    for ((key, value),(key2,value2)) in zip(model1.state_dict().items(), model2.state_dict().items()):
        res[key] = value + value2

    return res

def get_average_model(model_list):
    res = OrderedDict()
    for model in model_list:
        for k, v in model.state_dict().items():
            if k in res.keys():
                res[k] += v
            else:
                res[k] = v
    
    length = len(model_list)
    for k,v in res.items():
        v = torch.div(v, length)
    return res

def weighted_aggregate_model(model_list, weight_list):
    res = OrderedDict()
    for (w, model) in zip(weight_list, model_list):
        for k, v in model.state_dict():
            if k in res.keys():
                res[k] += w*v
            else:
                res[k] = w*v
    
    length = len(model_list)
    for k,v in res.items():
        v = torch.div(v, length)
    return res
