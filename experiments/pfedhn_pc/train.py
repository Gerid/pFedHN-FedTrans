import torch

from reformer_pytorch import Reformer, Recorder, LSHSelfAttention, FullQKAttention
from model import *


def inter_cluster_attn(cluster_list):
    
    input_q, input_v =[], []

    for cluster in cluster_list:
       input_q.append(cluster.emb_vec)
       input_v.append(cluster.per_layer)
    _, attn_matrix, _  = FullQKAttention(input_q, input_v)
    return attn_matrix
