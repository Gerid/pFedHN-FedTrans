from typing import OrderedDict
import torch
from torch import nn

from utils import *
from reformer_pytorch import Reformer, Recorder
from kmeans import KMeans
from train import inter_cluster_attn
"""
Client, the unit of federated learning
each stores its own data(implemented by DataLoader)


"""
class Client():
    def __init__(self, id, base_layer, base_optimizer, optimizer_config, device, cluster_per_layer=None, cluster=None):
        # local model
        self.base_layer = base_layer
        self.base_optimzer = base_optimizer
        self.per_layer = cluster_per_layer
        self.per_optimizer = base_optimizer(self.per_layer.parameters(), **optimizer_config)

        self.id = id #string
        self.cluster_id = -1 # default:-1, before form of cluster
        self.train_data = {}
        self.client_emb_vec = []
    
    def train():
        pass

    def emb_params():
        embed_vec()
    

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

    def pre_update(self):
        for i in range(self.pre_update_eps):
            self.assign_clients()
            for i in range(self.num_clients):
                self.base_layer = self.base_layer + self.client_list[i].base_layer.parameters()
                self.per_layer = self.per_layer + self.client_list[i].per_layer.parameters()
            self.base_layer = torch.div(self.base_layer, self.num_clients)
            self.per_layer = torch.div(self.per_layer, self.num_clients)
    
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

                  
            




    def add_client(self, client):
        self.client_list.append(client)
        self.assign_cluster(client)

    def assign_cluster(client):
        # TODO: add new client
        # cluster_similarity
        return client

            
    def assign_clients(self):
        for client in self.client_list:
            client.base_layer = self.base_layer
    
    def client_update(self):
        base_update = []
        per_update = []

        for client in self.client_list:
            i_base_update, i_per_update = client.local_train() #client.local_train() returns w_base_layer, w_per_layer
            base_update.append(i_base_update)
            per_update.append(i_per_update)
        return base_update, per_update
    
    
class LSH_Attention():
    model = Reformer(
        dim = 512,
        depth=12,
        max_seq_len=8192,
        heads=8,
        lsh_dropout=0.1,
        causal=True
    ).cuda()


    model = Recorder(model)

    x = torch.randn(1, 8192, 512).cuda()
    y = model(x)

    # a list of attention weights and buckets for the first forward pass
    model.recordings[0]

    model.turn_off()  # stop recording
    model.turn_on()  # start recording
    model.clear()  # clear the recordings

    model = model.eject()  # recover the original model and remove all listeners


class Params_embed_layer(nn.Module):

    def __init__(self, input_dim, emb_dim):
        super(Params_embed_layer, self).__init__()
        self.embed = nn.Linear(input_dim, emb_dim)

    def forward(self, x):
        x = self.embed(x)
        return x
