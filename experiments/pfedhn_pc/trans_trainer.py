import argparse
import json
import logging
import random
from collections import defaultdict, OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from experiments.pfedhn_pc.utils import get_average_model, weighted_aggregate_model
from tqdm import trange

from experiments.pfedhn_pc.models import CNNHyperPC, CNNTargetPC, CNNTargetPC_M, LocalLayer
from experiments.pfedhn_pc.node import BaseNodesForLocal, BaseNodesForLocals_M
from experiments.utils import get_device, set_logger, set_seed, str2bool

import copy
from model import *
from utils import get_model

def eval_model(nodes, num_nodes, net, criteria, device, split):
    curr_results = evaluate(nodes, num_nodes, net, criteria, device, split=split)
    total_correct = sum([val['correct'] for val in curr_results.values()])
    total_samples = sum([val['total'] for val in curr_results.values()])
    avg_loss = np.mean([val['loss'] for val in curr_results.values()])
    avg_acc = total_correct / total_samples

    all_acc = [val['correct'] / val['total'] for val in curr_results.values()]

    return curr_results, avg_loss, avg_acc, all_acc


@torch.no_grad()
def evaluate(nodes: BaseNodesForLocal, num_nodes, net, criteria, device, split='test'):
    net.eval()
    results = defaultdict(lambda: defaultdict(list))

    for node_id in range(num_nodes):  # iterating over nodes
        running_loss, running_correct, running_samples = 0., 0., 0.
        if split == 'test':
            curr_data = nodes.test_loaders[node_id]
        elif split == 'val':
            curr_data = nodes.val_loaders[node_id]
        else:
            curr_data = nodes.train_loaders[node_id]


        for batch_count, batch in enumerate(curr_data):
            img, label = tuple(t.to(device) for t in batch)
            net_out = net(img)
            pred = nodes.models[node_id](net_out)
            running_loss += criteria(pred, label).item()
            running_correct += pred.argmax(1).eq(label).sum().item()
            running_samples += len(label)

        results[node_id]['loss'] = running_loss / (batch_count + 1)
        results[node_id]['correct'] = running_correct
        results[node_id]['total'] = running_samples

    return results


def train(data_name: str, data_path: str, classes_per_node: int, num_nodes: int,
          steps: int, inner_steps: int, optim: str, lr: float, inner_lr: float,
          embed_lr: float, wd: float, inner_wd: float, embed_dim: int, hyper_hid: int,
          n_hidden: int, n_kernels: int, bs: int, device, eval_every: int, save_path: Path,
          net_args: dict, # config for inet
          step_attn_avg: int, #epochs for attentive aggregation
          ) -> None:

    ###############################
    # init nodes, inet, local net #
    ###############################


    nodes = BaseNodesForLocals_M(
        data_name=data_name,
        data_path=data_path,
        n_nodes=num_nodes,
        base_layer=CNNTargetPC,
        layer_config=None,
        base_optimizer=torch.optim.SGD, optimizer_config=dict(lr=inner_lr, momentum=.9, weight_decay=inner_wd),
        device=device,
        batch_size=bs,
        classes_per_node=classes_per_node,
    )

    embed_dim = embed_dim
    if embed_dim == -1:
        logging.info("auto embedding size")
        embed_dim = int(1 + num_nodes / 4)

    inet = CNNTargetPC_M()
    #net = CNNTargetPC(n_kernels=n_kernels)

    inet = inet.to(device)
    #net = net.to(device)#unnecessary

    ##################
    # init optimizer #
    ##################
    embed_lr = embed_lr if embed_lr is not None else lr
    optimizers = {
        'sgd': torch.optim.SGD(
            [
                {'params': [p for n, p in inet.named_parameters() if 'embed' not in n]},
                {'params': [p for n, p in inet.named_parameters() if 'embed' in n], 'lr': embed_lr}
            ], lr=lr, momentum=0.9, weight_decay=wd
        ),
        'adam': torch.optim.Adam(params=inet.parameters(), lr=lr)
    }
    optimizer = optimizers[optim]
    criteria = torch.nn.CrossEntropyLoss()

    ################
    # init metrics #
    ################
    last_eval = -1
    best_step = -1
    best_acc = -1
    test_best_based_on_step, test_best_min_based_on_step = -1, -1
    test_best_max_based_on_step, test_best_std_based_on_step = -1, -1
    step_iter = trange(steps)

    results = defaultdict(list)

    net_keys = [*inet.state_dict().keys()]
    base_layer_keys = net_keys[:-2]
    per_layer_keys = net_keys[-2:]

    net_values = [*inet.state_dict().values()]
    base_values = net_values[:-2]
    per_values = net_values[-2:]

    server = Server(base_values, num_client=100, pre_update_eps=100, per_layer=per_values, num_cluster=10)
    for step in step_iter:
        inet.train()

        # each client load global weights
        nodes.client_load_weights(inet)
        # select client at random
        node_id = random.choice(range(num_nodes))

        # NOTE: evaluation on sent model
        with torch.no_grad():
            for n in nodes.models:
                n.eval()
            inet.eval()
            batch = next(iter(nodes.test_loaders[node_id]))
            img, label = tuple(t.to(device) for t in batch)

            pred = nodes.model[node_id](img)

            prvs_loss = criteria(pred, label)
            prvs_acc = pred.argmax(1).eq(label).sum().item() / len(label)
            inet.train()

        # inner updates -> obtaining theta_tilda
        for i in range(inner_steps):
            for n in nodes.model:
                n.train()
            nodes.local_optimizers[node_id].zero_grad()

            batch = next(iter(nodes.train_loaders[node_id]))
            img, label = tuple(t.to(device) for t in batch)

            pred = nodes.models[node_id](img)

            loss = criteria(pred, label)
            loss.backward()

            nodes.local_optimizers[node_id].step()

        for w_local in nodes.models:
            if w_glob is None:
                w_glob = copy.deepcopy(w_local)
            else:
                for k in w_glob.keys():
                    w_glob[k] += w_local[k]
        
        inet.load_state_dict(w_glob)


        step_iter.set_description(
            f"Step: {step+1}, Node ID: {node_id}, Loss: {prvs_loss:.4f},  Acc: {prvs_acc:.4f}"
        )

        if step % eval_every == 0:
            last_eval = step
            step_results, avg_loss, avg_acc, all_acc = eval_model(
                nodes, num_nodes, inet, net, criteria, device, split="test"
            )
            logging.info(f"\nStep: {step+1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")

            results['test_avg_loss'].append(avg_loss)
            results['test_avg_acc'].append(avg_acc)

            _, val_avg_loss, val_avg_acc, _ = eval_model(nodes, num_nodes, inet, net, criteria, device, split="val")
            if best_acc < val_avg_acc:
                best_acc = val_avg_acc
                best_step = step
                test_best_based_on_step = avg_acc
                test_best_min_based_on_step = np.min(all_acc)
                test_best_max_based_on_step = np.max(all_acc)
                test_best_std_based_on_step = np.std(all_acc)

            results['val_avg_loss'].append(val_avg_loss)
            results['val_avg_acc'].append(val_avg_acc)
            results['best_step'].append(best_step)
            results['best_val_acc'].append(best_acc)
            results['best_test_acc_based_on_val_beststep'].append(test_best_based_on_step)
            results['test_best_min_based_on_step'].append(test_best_min_based_on_step)
            results['test_best_max_based_on_step'].append(test_best_max_based_on_step)
            results['test_best_std_based_on_step'].append(test_best_std_based_on_step)

    if step != last_eval:
        _, val_avg_loss, val_avg_acc, _ = eval_model(nodes, num_nodes, inet, net, criteria, device, split="val")
        step_results, avg_loss, avg_acc, all_acc = eval_model(nodes, num_nodes, inet, net, criteria, device, split="test")
        logging.info(f"\nStep: {step + 1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")

        results['test_avg_loss'].append(avg_loss)
        results['test_avg_acc'].append(avg_acc)

        if best_acc < val_avg_acc:
            best_acc = val_avg_acc
            best_step = step
            test_best_based_on_step = avg_acc
            test_best_min_based_on_step = np.min(all_acc)
            test_best_max_based_on_step = np.max(all_acc)
            test_best_std_based_on_step = np.std(all_acc)

        results['val_avg_loss'].append(val_avg_loss)
        results['val_avg_acc'].append(val_avg_acc)
        results['best_step'].append(best_step)
        results['best_val_acc'].append(best_acc)
        results['best_test_acc_based_on_val_beststep'].append(test_best_based_on_step)
        results['test_best_min_based_on_step'].append(test_best_min_based_on_step)
        results['test_best_max_based_on_step'].append(test_best_max_based_on_step)
        results['test_best_std_based_on_step'].append(test_best_std_based_on_step)

    emb_layer = nn.Linear(len(nn.utils.parameters_to_vector(per_values)), 128)
    #embed client
    emb_vectors = []
    for m in nodes.models:#parallalise
        net_values = [*m.state_dict().values()]
        per_values = net_values[-2:]
        emb_vectors.append(emb_layer(nn.utils.parameters_to_vector(per_values)))

    for step in step_attn_avg: 
        server.client_emb_list = emb_vectors
        server.form_cluster()
        least_cluster_len = 0
        for cl in server.cluster_list:
            least_cluster_len = min(least_cluster_len, len(cl))
        #local updates
        for i in range(inner_steps):
            for n in nodes.model:
                n.train()

            nodes.local_optimizers[node_id].zero_grad()

            batch = next(iter(nodes.train_loaders[node_id]))
            img, label = tuple(t.to(device) for t in batch)

            pred = nodes.models[node_id](img)

            loss = criteria(pred, label)
            loss.backward()

            nodes.local_optimizers[node_id].step()

        avg_model = get_average_model(nodes.models) 
        inet.load_state_dict(avg_model)

        #attention input: seq of emb_cluster and emb_clients_incluster
        emb_vectors = []
        for m in nodes.models:#parallalise
            net_values = [*m.state_dict().values()]
            per_values = net_values[-2:]
            emb_vectors.append(emb_layer(nn.utils.parameters_to_vector(per_values)))

        # get cluster embedding
        # cluster_emb = avg(client_emb)
        c_emb_list = []
        for cluster in server.cluster_list:
            for c in cluster.client_list:
                if emb_v_c is None:
                    emb_v_c = emb_vectors[c.id]
                else:
                    emb_v_c += emb_vectors[c.id]
            c_emb = torch.div(emb_v_c, len(c))
            c_emb_list.append(c_emb)

        cluster_num = len(server.cluster_list)
        c_model_list = [c.model for c in server.cluster_list]

        #nodes aggregation
        inter_attn = LSH_Attention(
            dim = 128,
            heads = 8,
            bucket_size = cluster_num / 2, #seqlen % (bucket_size * 2) == 0
            n_hashes = 8,
            return_attn = True
        )

        # inter-cluste-attn-aggregation
        # unsqueeze list to (batch_size=1, seqlen, dim)
        # fit input size for attn
        _, inter_attn_mat, _ = inter_attn(c_emb_list.unsqueeze(0))
        for i in range(cluster_num):
            weight_list = inter_attn_mat[0,i,:]
            server.cluster_list[i].model = weighted_aggregate_model(c_model_list, weight_list)


        # intra-cluster-attn-aggregation
        # c_emb_vec stands for cluster embedding vector
        for cluster in server.cluster_list:
            cluster_emb_list = []
            # least_cluster_len for shortest cluster 
            # for better design, we should fix seq_len of participated client
            # padding with zeros
            # during attention period, we just mask them out
            intra_seq_len = 10
            pad_cluster = False
            # random choice may cause duplicated?
            idx_list = [random.choice(range(len(cluster.client_list))) for _ in range(intra_seq_len)]
            client_model_list = []
            client_grad_list = []
            for i in idx_list:
                c_id = cluster.client_list[i].id
                c_emb_vec = emb_vectors[c_id]    
                cluster_emb_list.append(c_emb_vec)
                client_model_list.append(nodes.models[c.id])
                # get clients.grad
                client_grad_list.append(nodes.models[c.id].grad)

            _, intra_attn_mat, _ = LSH_Attention(
                dim = 128,
                heads = 8,
                bucket_size = intra_seq_len, #seqlen % (bucket_size * 2) == 0
                n_hashes = 8,
                return_attn = True
            )

            # model aggregation or grad aggregation?
            # some params needs tobe detach()
            for i, c_id in enumerate(idx_list):
                weight_list = intra_attn_mat[0,i,:]
                state_dict = add2model(cluster.model, weighted_aggregate_model(client_grad_list, weight_list))
                cluster.client_list[c_id].load_state_dict(state_dict) 
            


            



        
        
        

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(str(save_path / "results.json"), "w") as file:
        json.dump(results, file, indent=4)


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
    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    set_logger()
    set_seed(args.seed)

    device = get_device(gpus=args.gpu)

    if args.data_name == 'cifar10':
        args.classes_per_node = 2
    else:
        args.classes_per_node = 10

    train(
        data_name=args.data_name,
        data_path=args.data_path,
        classes_per_node=args.classes_per_node,
        num_nodes=args.num_nodes,
        steps=args.num_steps,
        inner_steps=args.inner_steps,
        optim=args.optim,
        lr=args.lr,
        inner_lr=args.inner_lr,
        embed_lr=args.embed_lr,
        wd=args.wd,
        inner_wd=args.inner_wd,
        embed_dim=args.embed_dim,
        hyper_hid=args.hyper_hid,
        n_hidden=args.n_hidden,
        n_kernels=args.nkernels,
        bs=args.batch_size,
        device=device,
        eval_every=args.eval_every,
        save_path=args.save_path,
    )
