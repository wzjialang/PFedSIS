import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn


@torch.no_grad()
def update_global_model_with_channels_qkv(net_clients, keys, shape_similarity_clients):
    client_num = len(net_clients)
    if len(net_clients) == 3:
        iter_container = zip(
            net_clients[0].named_parameters(),
            net_clients[1].named_parameters(),
            net_clients[2].named_parameters(),
        )
    elif len(net_clients) == 4:
        iter_container = zip(
            net_clients[0].named_parameters(),
            net_clients[1].named_parameters(),
            net_clients[2].named_parameters(),
            net_clients[3].named_parameters(),
        )
    elif len(net_clients) == 6:
        iter_container = zip(
            net_clients[0].named_parameters(),
            net_clients[1].named_parameters(),
            net_clients[2].named_parameters(),
            net_clients[3].named_parameters(),
            net_clients[4].named_parameters(),
            net_clients[5].named_parameters(),
        )
    else:
        iter_container = zip(net_clients[0].named_parameters())

    for data in iter_container:
        name = [d[0] for d in data]
        param = [d[1] for d in data]
        merge_shape_similarity = torch.stack([shape_similarity_clients[i][name[0]] for i in range(client_num)], dim=0)
        shape_similarity_matrix = torch.nn.functional.softmax(merge_shape_similarity, dim=0)
        if "attn.q" in name[0]:
            c = param[0].shape[0] // 2
            new_para = Variable(torch.Tensor(np.zeros(param[0].shape)), requires_grad=False).cuda()
            for i in range(client_num):
                new_para[:c, ...] += param[i][:c, ...] * shape_similarity_matrix[i][:c, ...]
            for i in range(client_num):
                param[i][:c, ...].data.mul_(0).add_(new_para[:c, ...].data)
        elif "attn.kv" in name[0]:
            c = param[0].shape[0] // 4
            new_para = Variable(torch.Tensor(np.zeros(param[0].shape)), requires_grad=False).cuda()
            for i in range(client_num):
                new_para[2 * c : 3 * c, ...] += param[i][2 * c : 3 * c, ...] * shape_similarity_matrix[i][2 * c : 3 * c, ...]
                new_para[:c, ...] += param[i][:c, ...] * shape_similarity_matrix[i][:c, ...]
            for i in range(client_num):
                param[i][2 * c : 3 * c, ...].data.mul_(0).add_(new_para[2 * c : 3 * c, ...].data)
                param[i][:c, ...].data.mul_(0).add_(new_para[:c, ...].data)
        elif not name[0] in keys:
            continue
        else:
            new_para = Variable(torch.Tensor(np.zeros(param[0].shape)), requires_grad=False).cuda()
            for i in range(client_num):
                new_para += param[i].data * shape_similarity_matrix[i]
            for i in range(client_num):
                param[i].data.mul_(0).add_(new_para.data)
