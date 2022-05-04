import argparse
import json
import logging
import random
from collections import defaultdict, OrderedDict
from pathlib import Path

import copy
import numpy as np
import torch
import torch.utils.data
from tqdm import trange

from experiments.pfedhn_pc.models import CNNHyperPC, CNNTargetPC, LocalLayer
from experiments.pfedhn_pc.node import BaseNodesForLocal
from experiments.utils import get_device, set_logger, set_seed, str2bool

from update import LocalUpdate
from kmeans import KMeans

import torch.nn.functional as F




def train(data_name: str, data_path: str, classes_per_node: int, num_nodes: int,
          steps: int, inner_steps: int, optim: str, lr: float, inner_lr: float,
          embed_lr: float, wd: float, inner_wd: float, embed_dim: int, hyper_hid: int,
          n_hidden: int, n_kernels: int, bs: int, device, eval_every: int, save_path: Path,
          args: dict, input_dim: int, params_embed_layer# not original param
          
          ) -> None:

    ###############################
    # init nodes, hnet, local net #
    ###############################

    nodes = BaseNodesForLocal(
        data_name=data_name,
        data_path=data_path,
        n_nodes=num_nodes,
        base_layer=LocalLayer,
        layer_config={'n_input': 84, 'n_output': 10 if data_name == 'cifar10' else 100},
        base_optimizer=torch.optim.SGD, optimizer_config=dict(lr=inner_lr, momentum=.9, weight_decay=inner_wd),
        device=device,
        batch_size=bs,
        classes_per_node=classes_per_node,
        
        
    )

    embed_dim = embed_dim
    if embed_dim == -1:
        logging.info("auto embedding size")
        embed_dim = int(1 + num_nodes / 4)


    # normal fedavg procedure and get base layer and personalized layer.
    # for t_0 rounds.
    # output BaseLayer, PerLayer
    net_glob = get_model(args)
    net_glob.train(avg_steps)

    clients = []

    params = {
        'avg_steps': 10, # number of fedavg steps to obtain global model
        'num_users': 100 # number of clients
    }

    lens = []
    # build model
    net_glob = get_model(args)
    net_glob.train(params["avg_steps"])

    # NOTE:further embedding graph
    total_num_layers = len(net_glob.state_dict().keys)
    print(net_glob.state_dict().keys())
    net_keys = [*net_glob.state_dict().keys()]

    # specify representation parameters (in w_glob_keys) and head parameters (all others)
    # split the model into base layer and personalized layer.
    if args.alg == 'fedtrans':
        if 'cifar' in args.dataset:
            w_glob_keys = net_keys[:-2]
            w_per_keys = net_keys[-1:]

    print(total_num_layers)
    print(w_glob_keys)
    print(net_keys)
    
    if args.alg == 'fedtrans':
        num_param_glob = 0
        num_param_local = 0
        for key in net_glob.state_dict().keys():
            num_param_local += net_glob.state_dict()[key].numel()
            print(num_param_local)
            if key in w_glob_keys:
                num_params_glob += net_glob.state_dict()[key].numel()
        percentage_param = 100 * float(num_param_glob) / num_param_local
        print('# Params: {:.2f} (local), {:.2f} (global); Percentage {:.2f} ({}/{})'.format(num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))
    print("learning rate, batch size: {}, {}".format(args.lr, args.local_bs)

    # generate list of local models for each user
    w_locals_base = {}
    w_locals_per = {}
    for user in range(args['num_users']):
        w_local_dict_base = {}
        w_local_dict_per = {}
        for key in net_glob.state_dict().keys():
            if key in w_glob_keys:
                w_local_dict_base[key] = net_glob.state_dict()[key]

            if key in w_per_keys:
                w_local_dict_per[key] = net_glob.state_dict()[key]
        w_locals_base[user] = w_local_dict_base
        w_locals_per[user] = w_local_dict_per

    
    
    # training
    # initially, FedAvg t rounds get base model
    indd = None # indices of embedding for sentl140
    loss_train = []
    accs = []
    times = []
    accs10 = 0 
    accs10_glob = 0
    start = time.time()
    # avg_epochs iterations to get pre-trained model
    for iter in range(args.avg_epochs+1):
        w_glob = {}
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        if iter == args.epochs:
            m = args.num_users
        # idxs_users = np.random.choice(range(args.num_users))
        idx_users = range(args.num_users)# w/o random choice, fully participation
        w_keys_epoch_glob = w_glob_keys_glob
        w_keys_epoch_per = w_glob_keys_per
        time_in = []
        total_len = 0
        for ind, idx in enumerate(idxs_users):
            start_in = time.time()
            # if 'femnist' in args.dataset 
            alg = args.alg
            args.alg = 'fedavg'
            if args.avg_epochs == iter:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_ft])# NOTE：take care
            else:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_tr])
            args.alg = alg

            net_local = copy.deepcopy(net_glob)
            w_local = net_local.state_dict()

            net_local.load_state_dict(w_local)
            last = iter == args.epochs
            # if 'femnist'
            w_local, loss, indd = local.train(net=net_local.to(args.device), idx=idx, w_glob_keys=w_glob_keys, lr=args.lr, last=last)
            loss_local.append(copy.deepcopy(loss))
            total_len += lens[idx]

            if len(w_glob) == 0:
                w_glob = copy.deepcopy(w_local)
                for k, key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] = w_glob[key] * lens[idx]
                    w_local[idx][key] = w_local[key]
            else:
                for k, key in enumerate(net_glob.state_dict().keys():
                    if key in w_glob_keys:
                        w_glob[key] += w_local[key] *lens[idx]
                    else:
                        w_glob[key] += w_local[key]*lens[idx]
                    w_locals[idx][key] = w_local[key]
            
            times_in.append( time.time() - start_in)
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # get weighted average for global weights
        for k in net_glob.state_dict().keys():
            w_glob[k] = torch.div(w_glob[k], total_len)

        w_local = net_glob.state_dict()
        for k in w_glob.keys():
            w_local[k] = w_glob[k]
        if args.epochs != iter:
            net_glob.load_state_dict(w_glob)
        
        # record training loss
        # if iter % args.test_freq==args.test_freq-1 or iter>=args.
                
    # second step, split local model
    for key in net_glob.state_dict().keys():
        if key in w_glob_keys:
            w_local_dict_base[key] = net_glob.state_dict()[key]

        if key in w_per_keys:
            w_local_dict_per[key] = net_glob.state_dict()[key]
    w_locals_base[user] = w_local_dict_base
    w_locals_per[user] = w_local_dict_per

    if input_dim is None:
        input_dim = len(w_locals_per[0].values())
    
    if params_embed_layer is None:
        params_embed_layer = nn.Linear(input_dim, args.output_dim)

    #need revise
    w_per_embvec = {}
    w_per_embvec[user] = params_embed_layer(nn.utils.parameters_to_vector(w_locals_per[user]))
        

        
    for iter in range(args.max_eps):
        if iter % args.recluster_eps == 0:
            kmeans = KMeans(n_clusters=10, mode='euclidean', verbose=1)
            cluster = kmeans.fit_predict(w_per_embvec)
            
            inter_cluster_attention(cluster)
        intra_cluster_attention(cluster)
        






    # now we are in the first 'trans' communication round.
    # we're gonna process the models' params for easy communication and computation
    
    # NOTE:consider compress gradient to improve efficiency
    # torch; DDP communication hooks 

    # 1. get the model params tensor[], normalize and sparse(drop some params)
    # embedding model params to 'd' dimension
    PerVec = embed(PerLayer, dimension=embed_dim) 

    # 2. use LSH algorithm to cluster the models.
    # idx = {1, 2,..., K}
    # return K different cluster center model(perlayer)
    clusters = Cluster(PerVec)

    # 3. cluster center attention
    # calculate attention matrix and update center
    clusters = Attention(clusters)

    # 4. inner-cluster attention
    for c in clusters:
        c = Attention(c)
    
    # 5. 

#nn.utils
#get model according to input args
def get_model(args):
    if args.model == 'cnn' and 'cifar100' in args.dataset:
        net_glob = CNNCifar100(args=args).to(args.device)
    print(net_glob)
    return net_glob


#network arch
#config nn arch with args
class CNNCifar100(nn.Module):
    def __init__(self, args):
        super(CNNCifar100, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.6)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, args.num_classes)
        self.cls = args.num_classes

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.drop((F.relu(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Generic local update class, implements local updates for FedRep, FedPer, LG-FedAvg, FedAvg, FedProx
class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        if 'femnist' in args.dataset or 'sent140' in args.dataset: 
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])),name=self.args.dataset), batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
         
        if 'sent140' in self.args.dataset and indd == None:
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        elif indd is not None:
            self.indd = indd
        else:
            self.indd=None        
        
        self.dataset=dataset
        self.idxs=idxs

    def train(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.1):
        bias_p=[]
        weight_p=[]
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
        [     
            {'params': weight_p, 'weight_decay':0.0001},
            {'params': bias_p, 'weight_decay':0}
        ],
        lr=lr, momentum=0.5
        )
        if self.args.alg == 'prox':
            optimizer = FedProx.FedProx(net.parameters(),
                             lr=lr,
                             gmf=self.args.gmf,
                             mu=self.args.mu,
                             ratio=1/self.args.num_users,
                             momentum=0.5,
                             nesterov = False,
                             weight_decay = 1e-4)

        local_eps = self.args.local_ep
        if last:
            if self.args.alg =='fedavg' or self.args.alg == 'prox':
                local_eps= 10
                net_keys = [*net.state_dict().keys()]
                if 'cifar' in self.args.dataset:
                    w_glob_keys = [net.weight_keys[i] for i in [0,1,3,4]]
                elif 'sent140' in self.args.dataset:
                    w_glob_keys = [net_keys[i] for i in [0,1,2,3,4,5]]
                elif 'mnist' in args.dataset:
                    w_glob_keys = [net_glob.weight_keys[i] for i in [0,1,2]]
            elif 'maml' in self.args.alg:
                local_eps = 5
                w_glob_keys = []
            else:
                local_eps =  max(10,local_eps-self.args.local_rep_ep)
        
        head_eps = local_eps-self.args.local_rep_ep
        epoch_loss = []
        num_updates = 0
        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(self.args.local_bs)
        for iter in range(local_eps):
            done = False

            if(iter < self.args.glob_avg_eps):
                
            # for FedRep, first do local epochs for the head
            if (iter < head_eps and self.args.alg == 'fedrep') or last:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            
            # then do local epochs for the representation
            elif iter == head_eps and self.args.alg == 'fedrep' and not last:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            # all other methods update all parameters simultaneously
            elif self.args.alg != 'fedrep':
                for name, param in net.named_parameters():
                     param.requires_grad = True 
       
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if 'sent140' in self.args.dataset:
                    input_data, target_data = process_x(images, self.indd), process_y(labels,self.indd)
                    if self.args.local_bs != 1 and input_data.shape[0] != self.args.local_bs:
                        break
                    net.train()
                    data, targets = torch.from_numpy(input_data).to(self.args.device), torch.from_numpy(target_data).to(self.args.device)
                    net.zero_grad()
                    hidden_train = repackage_hidden(hidden_train)
                    output, hidden_train = net(data, hidden_train)
                    loss = self.loss_func(output.t(), torch.max(targets, 1)[1])
                    loss.backward()
                    optimizer.step()
                else:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                num_updates += 1
                batch_loss.append(loss.item())
                if num_updates == self.args.local_updates:
                    done = True
                    break
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            if done:
                break
            
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd

def fedavg(client, totalround):
    #TODO
    return client


def embed(model_params, dimension=100):
    #TODO
    return model_params

class Cluster():
    def __init__(self) -> None:
        pass

    def change():
        pass

class Attention():
    def __init__(self) -> None:
        pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Federated Hypernetwork with local layers experiment"
    )

    #############################
    #       Dataset Args        #
    #############################
    parser.add_argument(
        "--data-name", type=str, default="cifar10", choices=['cifar10', 'cifar100'], help="data name"
    )
    parser.add_argument("--data-path", type=str, default='data', help='data path')
    parser.add_argument("--num-nodes", type=int, default=50)

    ##################################
    #       Optimization args        #
    ##################################
    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--inner-steps", type=int, default=50, help="number of inner steps")
    parser.add_argument("--optim", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")

    ################################
    #       Model Prop args        #
    ################################
    parser.add_argument("--n-hidden", type=int, default=3, help="num. hidden layers")
    parser.add_argument("--inner-lr", type=float, default=5e-3, help="learning rate for inner optimizer")
    parser.add_argument("--lr", type=float, default=5e-2, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--inner-wd", type=float, default=5e-5, help="inner weight decay")
    parser.add_argument("--embed-dim", type=int, default=-1, help="embedding dim")
    parser.add_argument("--embed-lr", type=float, default=None, help="embedding learning rate")
    parser.add_argument("--hyper-hid", type=int, default=100, help="hypernet hidden dim")
    parser.add_argument("--spec-norm", type=str2bool, default=False, help="hypernet hidden dim")
    parser.add_argument("--nkernels", type=int, default=16, help="number of kernels for cnn model")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval-every", type=int, default=30, help="eval every X selected epochs")
    parser.add_argument("--save-path", type=str, default="pfedhn_pc_cifar_res", help="dir path for output file")
    parser.add_argument("--seed", type=int, default=42, help="seed value")

    args = parser.parse_args()
