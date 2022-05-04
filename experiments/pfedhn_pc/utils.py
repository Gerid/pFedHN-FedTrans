import torch
from model import *

def create_client(server=server, num_client=num_client, **args):
    start_id = server.num_client
    for id in range(start_id+1, start_id+num_client):
        args['id'] = id
        client = Client(**args)
        server.client_list.append(client)
        client.base_layer = server.base_layer
        client.per_layer = server.per_layer


        
def LocalUpdate(server):
    
