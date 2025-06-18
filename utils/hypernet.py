import os
import pickle
from collections import OrderedDict
from typing import Dict, List, OrderedDict, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


class Linear(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        nn.init.uniform_(self.weight)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class HyperNetwork(nn.Module):
    def __init__(
        self,
        temp_path,
        embedding_dim: int,
        client_num: int,
        hidden_dim: int,
        backbone: nn.Module,
        gpu="cuda:0",
    ):
        super(HyperNetwork, self).__init__()
        self.device = gpu if torch.cuda.is_available() else "cpu"
        self.client_num = client_num
        self.embedding = nn.Embedding(client_num, embedding_dim, device=self.device)
        self.blocks_name = set([(".".join(n.split(".")[:3]) if "seg_blocks" in n else ".".join(n.split(".")[:2])) for n in backbone])
        print("Hypernetwork for:", self.blocks_name)
        self.cache_dir = temp_path + "hn"
        if not os.path.isdir(self.cache_dir):
            os.system(f"mkdir -p {self.cache_dir}")

        if os.listdir(self.cache_dir) != client_num:
            for client_id in range(client_num):
                with open(self.cache_dir + f"/{client_id}.pkl", "wb") as f:
                    pickle.dump(
                        {
                            "mlp": nn.Sequential(
                                nn.Linear(embedding_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(),
                            ),
                            # all negative tensor would be outputted sometimes if fc is torch.nn.Linear, which used kaiming init.
                            # so here use U(0,1) init instead.
                            "fc": {name: Linear(hidden_dim, client_num) for name in self.blocks_name},
                        },
                        f,
                    )
        # for tracking the current client's hn parameters
        self.current_client_id: int = None
        self.mlp: nn.Sequential = None
        self.fc_layers: Dict[str, Linear] = {}
        self.retain_blocks: List[str] = []

    def mlp_parameters(self) -> List[nn.Parameter]:
        return list(filter(lambda p: p.requires_grad, self.mlp.parameters()))

    def fc_layer_parameters(self) -> List[nn.Parameter]:
        params_list = []
        for block, fc in self.fc_layers.items():
            if block not in self.retain_blocks:
                params_list += list(filter(lambda p: p.requires_grad, fc.parameters()))
        return params_list

    def emd_parameters(self) -> List[nn.Parameter]:
        return list(self.embedding.parameters())

    def forward(self, client_id: int) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        self.current_client_id = client_id
        self.retain_blocks = []
        emd = self.embedding(torch.tensor(client_id, dtype=torch.long, device=self.device))
        self.load_hn()
        feature = self.mlp(emd)
        alpha = {block: F.relu(self.fc_layers[block](feature)) for block in self.blocks_name}

        return (
            alpha,
            self.retain_blocks,
        )

    def save_hn(self):
        for block, param in self.fc_layers.items():
            self.fc_layers[block] = param.cpu()
        with open(self.cache_dir + f"/{self.current_client_id}.pkl", "wb") as f:
            pickle.dump(
                {"mlp": self.mlp.cpu(), "fc": self.fc_layers},
                f,
            )
        self.mlp = None
        self.fc_layers = {}
        self.current_client_id = None

    def load_hn(self) -> Tuple[nn.Sequential, OrderedDict[str, Linear]]:
        with open(self.cache_dir + f"/{self.current_client_id}.pkl", "rb") as f:
            parameters = pickle.load(f)
        self.mlp = parameters["mlp"].to(self.device)
        for block, param in parameters["fc"].items():
            self.fc_layers[block] = param.to(self.device)

    def clean_models(self):
        if os.path.isdir(self.cache_dir):
            os.system(f"rm -rf {self.cache_dir}")
