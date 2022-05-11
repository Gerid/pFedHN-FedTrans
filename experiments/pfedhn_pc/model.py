from typing import OrderedDict
import torch
from torch import nn

from utils import *
from reformer_pytorch import Reformer, Recorder
from kmeans import KMeans
"""
Client, the unit of federated learning
each stores its own data(implemented by DataLoader)


"""
class Client():
    def __init__(self, c_id, base_layer, base_optimizer, optimizer_config, device, cluster_per_layer=None, cluster=None):
        # local model
        self.base_layer = base_layer
        self.base_optimzer = base_optimizer
        self.per_layer = OrderedDict()
        self.per_optimizer = base_optimizer(self.per_layer.parameters(), **optimizer_config)

        self.id = c_id #string
        self.cluster_id = -1 # default:-1, before form of cluster
        self.train_data = {}
        self.client_emb_vec = []
    

    

class Cluster():
    def __init__(self, cluster_model, c_id):
        self.cluster_id = c_id
        self.client_list = []# shape:[cluster_siz* cluster_size
        self.per_layer =  OrderedDict()

"""
Federated Server
Data:
    1. cluster_list
    2. 
Function:
    1. base_layer update(aggregate local updates, perform fedavg on it)
    2. communication 
"""
class Server():
    def __init__(self, base_layer, client_list, num_client, pre_update_eps, per_layer, num_cluster=10):
        self.cluster_list = [Cluster(c_id) for c_id in range(num_cluster)]# shape:[num_cluster] cluster instance
        self.num_clients = num_client
        self.client_list = client_list # [num_clients]*client instance
        self.base_layer = base_layer
        self.pre_update_eps = pre_update_eps
        self.per_layer = per_layer

    
    def base_aggregate(self):
        for i in range(self.num_clients):
            self.base_layer += self.client_list[i].base_layer.parameters()
        torch.div(self.base_layer, self.num_clients)

    
    def form_cluster(self):
        #compute similarity k-means
        kmeans = KMeans(n_clusters=self.num_cluster, mode='euclidean')
        client_emb_list = self.client_emb_list 

        cluster_res = kmeans.fit_predict(client_emb_list)
        for client_id, cluster_id in enumerate(cluster_res):
           self.client_list[client_id].cluster_id = cluster_res[client_id] 
           self.cluster_list[cluster_id].client_list.append(self.client_list[client_id])
        
        for cluster in self.cluster_list:
            model_list = []
            for client in cluster.client_list:
                model_list.append(client.per_layer())
            #cluster.per_layer is the centroid per_model for clients within cluster
            cluster.per_layer = get_average_model(model_list)

    
