import torch
from collections import OrderedDict
from model import *
import copy

def create_client(server=server, num_client=num_client, **args):
    start_id = server.num_client
    for id in range(start_id+1, start_id+num_client):
        args['id'] = id
        client = Client(**args)
        server.client_list.append(client)
        client.base_layer = server.base_layer
        client.per_layer = server.per_layer


        
    

def get_average_model(model_list):
    res = OrderedDict()
    for model in model_list:
        for k, v in model.state_dict():
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
