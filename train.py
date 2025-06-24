import os
from tensorboardX import SummaryWriter
import argparse
import time
import random
import numpy as np
from glob import glob
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import Dict, List, OrderedDict, Tuple, Union

from utils.summary import create_logger
from scripts.tester_utils_med import test_vis
from loss import LossMulti
from scripts.trainer_utils_med import update_global_model_with_channels_qkv
from net.Ours.pfedsis import BuildFPN
from utils.util import ForeverDataIterator
from utils.hypernet import HyperNetwork
from dataloaders.robotics_dataloader import Dataset
import torchvision.transforms as transforms
from dataloaders.transforms import *


def seed_torch(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_init_fn(num_workers, rank, seed):
    worker_seed = args.seed + num_workers * rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def calculate_shape_similarity(model, data_loader, max_iteration, means, std, client_idx):
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    shape_similarity = {n: p.clone().detach().fill_(0) for n, p in params.items()}

    model.eval()
    for i_batch in range(max_iteration):
        sampled_batch = next(data_loader)
        volume_batch = sampled_batch["image"][:, 0, :, :, :].cuda()

        if client_idx == 0:
            volume_batch_mix = mixstlye_image(volume_batch, means[1], std[1], means[2], std[2], beta)
        elif client_idx == 1:
            volume_batch_mix = mixstlye_image(volume_batch, means[0], std[0], means[2], std[2], beta)
        elif client_idx == 2:
            volume_batch_mix = mixstlye_image(volume_batch, means[0], std[0], means[1], std[1], beta)

        _, _, outputs_g = model(volume_batch)
        _, _, outputs_g_mix = model(volume_batch_mix)

        l2_pred = torch.mean(torch.pow(outputs_g, 2)) + torch.mean(torch.pow(outputs_g_mix, 2))

        model.zero_grad()
        l2_pred.backward()
        for n, p in shape_similarity.items():
            if params[n].grad is not None:
                p += params[n].grad.abs() / float(max_iteration)
    model.train()
    return shape_similarity


def mixstlye_image(content, mean1, std1, mean2, std2, beta, p=0.5):
    B = content.size(0)
    c_mean = torch.mean(content, [2, 3], keepdim=True)
    c_std = torch.std(content, [2, 3], keepdim=True)
    lmda = beta.sample((B, 1, 1, 1)).to(content.device)
    lmda1 = beta.sample((B, 1, 1, 1)).to(content.device)
    lmda2 = beta.sample((B, 1, 1, 1)).to(content.device)

    sum_lmda = lmda + lmda1 + lmda2
    lmda = lmda / sum_lmda.float()
    lmda1 = lmda1 / sum_lmda.float()
    lmda2 = lmda2 / sum_lmda.float()

    if random.random() > p:
        return content

    repeat_mean1 = torch.from_numpy(mean1).to(content.device).repeat(B, 1, 1, 1).permute(0, 3, 1, 2)
    repeat_mean2 = torch.from_numpy(mean2).to(content.device).repeat(B, 1, 1, 1).permute(0, 3, 1, 2)
    repeat_std1 = torch.from_numpy(std1).to(content.device).repeat(B, 1, 1, 1).permute(0, 3, 1, 2)
    repeat_std2 = torch.from_numpy(std2).to(content.device).repeat(B, 1, 1, 1).permute(0, 3, 1, 2)
    mu_mix = c_mean * lmda + repeat_mean1 * lmda1 + repeat_mean2 * lmda2
    sig_mix = c_std * lmda + repeat_std1 * lmda1 + repeat_std2 * lmda2

    return ((content - c_mean) / c_std) * sig_mix + mu_mix


def get_mean_std(dataset: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate dataset mean and standard deviation."""
    means = [0, 0, 0]
    std = [0, 0, 0]
    num_imgs = len(dataset)
    for _, sampled_batch in enumerate(dataset):
        img = sampled_batch["image"]
        for i in range(3):
            means[i] += img[:, :, i, :, :].mean()
            std[i] += img[:, :, i, :, :].std()

    means = np.asarray(means) / float(num_imgs)
    std = np.asarray(std) / float(num_imgs)

    print(f"normMean = {means}")
    print(f"normstd = {std}")
    return means, std


def generate_client_model_parameters(
    args,
    client_id: int,
    all_params_name: List[str],
    client_model_params_list: List[List[torch.Tensor]],
    hypernet: HyperNetwork,
    client_num_in_total: int,
    ignored: List[str],
) -> Tuple[OrderedDict[str, torch.Tensor], List[str], List[List[torch.Tensor]]]:
    """Generate client-specific model parameters using hypernetwork."""
    layer_params_dict = dict(zip(all_params_name, list(zip(*client_model_params_list))))
    alpha, retain_blocks = hypernet(client_id)
    aggregated_parameters = {}
    default_weight = torch.tensor(
        [i == client_id for i in range(client_num_in_total)],
        dtype=torch.float,
        # requires_grad=True,
    ).to(args.gpu)

    for name in all_params_name:
        if name in ignored:
            key = ".".join(name.split(".")[:3]) if "seg_blocks" in name else ".".join(name.split(".")[:2])
            a = alpha[key]
        else:
            a = default_weight

        if a.sum() == 0:
            raise RuntimeError(f"Client [{client_id}]'s {name.split('.')[0]} alpha is all zeros")

        aggregated_parameters[name] = torch.sum(
            a / a.sum() * torch.stack(layer_params_dict[name], dim=-1).cuda(),
            dim=-1,
        )

    client_model_params_list[client_id] = list(aggregated_parameters.values())
    return aggregated_parameters, retain_blocks, client_model_params_list


def clone_parameters(
    src: Union[OrderedDict[str, torch.Tensor], nn.Module],
) -> OrderedDict[str, torch.Tensor]:
    """Clone model parameters."""
    if isinstance(src, OrderedDict):
        return OrderedDict({name: param.clone().detach().requires_grad_(param.requires_grad) for name, param in src.items()})
    if isinstance(src, torch.nn.Module):
        return OrderedDict({name: param.clone().detach().requires_grad_(param.requires_grad) for name, param in src.state_dict(keep_vars=True).items()})


def update_hypernetwork(
    hypernet,
    ignored,
    client_model_params_list,
    client_id: int,
    diff: OrderedDict[str, torch.Tensor],
    retain_blocks: List[str] = [],
) -> None:
    # Calculate gradients
    hn_grads = torch.autograd.grad(
        outputs=list(
            filter(
                lambda param: param.requires_grad,
                client_model_params_list[client_id],
            )
        ),
        inputs=hypernet.mlp_parameters() + hypernet.fc_layer_parameters() + hypernet.emd_parameters(),
        grad_outputs=list(
            map(
                lambda tup: tup[1],
                filter(
                    lambda tup: tup[1].requires_grad and ".".join(tup[0].split(".")[:3]) not in retain_blocks and tup[0] in ignored,
                    diff.items(),
                ),
            )
        ),
        allow_unused=True,
    )
    mlp_grads = hn_grads[: len(hypernet.mlp_parameters())]
    fc_grads = hn_grads[len(hypernet.mlp_parameters()) : len(hypernet.mlp_parameters() + hypernet.fc_layer_parameters())]
    emd_grads = hn_grads[len(hypernet.mlp_parameters() + hypernet.fc_layer_parameters()) :]

    for param, grad in zip(hypernet.fc_layer_parameters(), fc_grads):
        if grad is not None:
            param.data -= 5e-3 * grad

    for param, grad in zip(hypernet.mlp_parameters(), mlp_grads):
        param.data -= 5e-3 * grad

    for param, grad in zip(hypernet.emd_parameters(), emd_grads):
        param.data -= 5e-3 * grad

    hypernet.save_hn()
    return hypernet


@torch.no_grad()
def update_client_model_parameters(
    client_model_params_list,
    client_id: int,
    delta: OrderedDict[str, torch.Tensor],
) -> None:
    updated_params = []
    for param, diff in zip(client_model_params_list[client_id], delta.values()):
        updated_params.append((param + diff).detach())
    client_model_params_list[client_id] = updated_params
    return client_model_params_list


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="pfedsis", help="model_name")
    parser.add_argument("--dataset", type=str, default="robotics", help="dataset name")
    parser.add_argument("--max_epoch", type=int, default=400, help="maximum epoch number to train")
    parser.add_argument("--batch_size", type=int, default=8, help="batch_size for each client")
    parser.add_argument(
        "--base_lr",
        type=float,
        default=0.0001,
        help="basic learning rate of each client",
    )
    parser.add_argument("--seed", type=int, default=2022, help="random seed")
    parser.add_argument("--gpu", type=str, default="cuda:0", help="GPU to use")
    parser.add_argument("--iteration", type=int, default=100, help="max iteration number")
    parser.add_argument("--num_classes", type=int, default=4, help="number of classes")
    parser.add_argument("--c_in", type=int, default=3, help="number of input channels")
    parser.add_argument("--client_num", type=int, default=3, help="number of clients")
    args = parser.parse_args()

    seed_torch(seed=args.seed)

    # Paths setup
    txt_path = f"./{args.dataset}/{args.exp}/txt/"
    log_path = f"./{args.dataset}/{args.exp}/log/"
    model_path = f"./{args.dataset}/{args.exp}/weight/"
    temp_path = f"./{args.dataset}/{args.exp}/temp/"
    os.makedirs(txt_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    logger = create_logger(0, save_dir=txt_path)
    print = logger.info

    assert args.num_classes > 0 and args.client_num > 1
    print(args)

    # ------------------  start training ------------------ #
    client_weight = np.ones((args.client_num,)) / args.client_num
    print(client_weight)

    dataloader_clients = []
    net_clients = []
    optimizer_clients = []
    dataloader_test_clients = []
    means_clients = []
    std_clients = []
    best_score = [0, 0, 0]
    best_score_iou = [0, 0, 0]
    loss_func = LossMulti(4, jaccard_weight=0.3)
    best_score_overall = 0
    best_score_iou_overall = 0
    total_loss_epoch = 0
    writer = SummaryWriter(log_path)
    l2loss = torch.nn.MSELoss()
    beta = torch.distributions.Beta(0.1, 0.1)
    loss_reg_fn = nn.MSELoss()
    train_transforms = transforms.Compose(
        [
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomScaleCrop(base_size={"h": 512, "w": 640}, crop_size={"h": 512, "w": 640}),
            ToTensor(),
        ]
    )
    test_transforms = transforms.Compose([ToTensor()])

    for client_idx in range(args.client_num):
        dataset = Dataset(client_idx=client_idx, split="train", transform=train_transforms)
        dataset_test = Dataset(client_idx=client_idx, split="test", transform=test_transforms)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=3,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn(0, 0, 0),
        )
        dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=3, pin_memory=True)
        means, std = get_mean_std(dataloader)
        means_clients.append(means)
        std_clients.append(std)

        dataloader = ForeverDataIterator(dataloader)
        dataloader_clients.append(dataloader)
        dataloader_test_clients.append(dataloader_test)

        net = BuildFPN(args.num_classes, "pvtb0", "fpn")
        net = net.cuda()
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.base_lr, betas=(0.9, 0.999))
        optimizer_clients.append(optimizer)
        net_clients.append(net)

    ignore_keys = [
        "attn.q",
        "attn.kv",
        "head_p",
        "decoder_p",
        "head_res",
        "decoder_p_res",
    ]
    global_keys = []
    ignored = []
    for k in net.state_dict().keys():
        ignore_tag = 0
        for ignore_key in ignore_keys:
            if ignore_key in k:
                ignore_tag = 1
        if not ignore_tag:
            global_keys.append(k)
        else:
            ignored.append(k)
    # print("Global keys:", global_keys)
    # print("Partial keys:", ignored)

    embedding_dim = 100
    hidden_dim = 100
    hypernet = HyperNetwork(
        temp_path=temp_path,
        embedding_dim=embedding_dim,
        client_num=args.client_num,
        hidden_dim=hidden_dim,
        backbone=ignored,
        gpu=args.gpu,
    ).to(args.gpu)

    all_params_name = [name for name in net.state_dict().keys()]
    if os.listdir(temp_path) != []:
        if os.path.exists(temp_path + "/clients_model.pt"):
            client_model_params_list = torch.load(temp_path + "/clients_model.pt")
            print("Find existed clients model...")
        else:
            print("Initializing clients model...")
            client_model_params_list = [list(net.state_dict().values()) for _ in range(args.client_num)]

    print("[INFO] Initialized success...")
    # Start federated learning
    for epoch_num in range(args.max_epoch):
        shape_similarity_clients = []
        for client_idx in range(args.client_num):
            dataloader_current = dataloader_clients[client_idx]
            net_current = net_clients[client_idx]
            net_current.train()
            optimizer_current = optimizer_clients[client_idx]
            time1 = time.time()

            # Update the personalized parameter of site m at the server via Eq. (7)
            (
                client_local_params,
                retain_blocks,
                client_model_params_list,
            ) = generate_client_model_parameters(
                args,
                client_idx,
                all_params_name,
                client_model_params_list,
                hypernet,
                args.client_num,
                ignored,
            )

            net_current.load_state_dict(client_local_params, strict=True)
            frz_model_params = clone_parameters(net_current)
            for i_batch in range(args.iteration):
                sampled_batch = next(dataloader_current)
                time2 = time.time()
                volume_batch, label_batch = (
                    sampled_batch["image"][:, 0, :, :, :],
                    sampled_batch["label"].cuda(),
                )
                volume_batch = volume_batch.cuda()
                volume_batch_mix = volume_batch.clone()

                # Cross-Style Shape Consistency
                if client_idx == 0:
                    volume_batch_mix = mixstlye_image(
                        volume_batch_mix,
                        means_clients[1],
                        std_clients[1],
                        means_clients[2],
                        std_clients[2],
                        beta,
                        p=2,
                    )
                elif client_idx == 1:
                    volume_batch_mix = mixstlye_image(
                        volume_batch_mix,
                        means_clients[0],
                        std_clients[0],
                        means_clients[2],
                        std_clients[2],
                        beta,
                        p=2,
                    )
                elif client_idx == 2:
                    volume_batch_mix = mixstlye_image(
                        volume_batch_mix,
                        means_clients[0],
                        std_clients[0],
                        means_clients[1],
                        std_clients[1],
                        beta,
                        p=2,
                    )

                outputs, outputs_res, _ = net_current(volume_batch)
                outputs_mix, outputs_res_mix, _ = net_current(volume_batch_mix)

                loss_seg = loss_func(outputs, label_batch)
                loss_res = loss_reg_fn(outputs_res, volume_batch)
                loss_seg_mix = loss_func(outputs_mix, label_batch)
                loss_res_mix = loss_reg_fn(outputs_res_mix, volume_batch_mix)
                total_loss = (loss_seg + loss_seg_mix + loss_res + loss_res_mix) / 2.0

                optimizer_current.zero_grad()
                total_loss.backward()
                optimizer_current.step()
                total_loss_epoch += total_loss.item()

                if i_batch % 10 == 0:
                    print(
                        "Epoch: [%d] site [%d] iteration [%d / %d] : total loss: %f, loss_seg: %f, loss_res: %f, loss_res_mix: %f, loss_seg_mix: %f"
                        % (
                            epoch_num,
                            client_idx + 1,
                            i_batch,
                            args.iteration,
                            total_loss.item(),
                            loss_seg.item(),
                            loss_res.item(),
                            loss_res_mix.item(),
                            loss_seg_mix.item(),
                        )
                    )

            total_loss_epoch /= float(args.iteration)
            writer.add_scalar(
                "loss_total_per_epoch/site{}".format(client_idx + 1),
                total_loss_epoch,
                epoch_num,
            )

            # Hypernetwork-Guided Update
            diff = OrderedDict(
                {
                    k: p1 - p0
                    for (k, p1), p0 in zip(
                        net_current.state_dict(keep_vars=True).items(),
                        frz_model_params.values(),
                    )
                }
            )
            hypernet = update_hypernetwork(
                hypernet,
                ignored,
                client_model_params_list,
                client_idx,
                diff,
                retain_blocks,
            )  # Update the hypernetwork HNm(νm;φm) via Eqs. (8)(9)
            client_model_params_list = update_client_model_parameters(client_model_params_list, client_idx, diff)  # Update each local site's model parameters via local iteration

            # Shape-Similarity Update
            shape_similarity_clients.append(
                calculate_shape_similarity(
                    net_current,
                    dataloader_clients[client_idx],
                    args.iteration,
                    means_clients,
                    std_clients,
                    client_idx,
                )
            )  # Calculate the shape-similarity via Eq. (13)
        update_global_model_with_channels_qkv(net_clients, global_keys, shape_similarity_clients)  # Eqs. (14)(15)

        for client_idx in range(args.client_num):
            client_model_params_list[client_idx] = list(net_clients[client_idx].state_dict().values())

        ## Evaluation
        if epoch_num >= 100:
            overall_score = 0
            overall_score_iou = 0
            for site_index in range(args.client_num):
                this_net = net_clients[site_index]
                dice_list = []
                print("[Test] epoch {} testing Site {}".format(epoch_num, site_index + 1))

                score_values = test_vis(
                    site_index,
                    args,
                    dataloader_test_clients[site_index],
                    this_net,
                    visulization=False,
                )

                writer.add_scalar(
                    "DICE_per_epoch/site{}".format(site_index + 1),
                    score_values[0],
                    epoch_num,
                )
                writer.add_scalar(
                    "IOU_per_epoch/site{}".format(site_index + 1),
                    score_values[1],
                    epoch_num,
                )

                if score_values[0] > best_score[site_index]:
                    best_score[site_index] = score_values[0]
                    best_score_iou[site_index] = score_values[1]
                    save_mode_path = os.path.join(model_path, "Site{}_best.pth".format(site_index + 1))

                    torch.save(this_net.state_dict(), save_mode_path)

                print(
                    "[INFO] Score/site{} Dice: {:.2f} IOU: {:.2f} Best Dice {:.2f} Best IOU {:.2f}".format(
                        site_index + 1,
                        score_values[0] * 100,
                        score_values[1] * 100,
                        best_score[site_index] * 100,
                        best_score_iou[site_index] * 100,
                    )
                )

                overall_score += score_values[0]
                overall_score_iou += score_values[1]

            overall_score /= args.client_num
            overall_score_iou /= args.client_num
            writer.add_scalar("Score_Overall/", overall_score, epoch_num)
            writer.add_scalar("Score_Overall_iou/", overall_score_iou, epoch_num)

            if overall_score > best_score_overall:
                best_score_overall = overall_score
                best_score_iou_overall = overall_score_iou

                ## save mode
                for site_index in range(args.client_num):
                    save_mode_path = os.path.join(model_path, "Overall_Site{}_best.pth".format(site_index + 1))
                    torch.save(net_clients[site_index].state_dict(), save_mode_path)
            print(
                "[INFO] Dice Overall: {:.2f}  IOU Overall: {:.2f} Best Dice Overall: {:.2f} Corr IOU Overall: {:.2f}".format(
                    overall_score * 100,
                    overall_score_iou * 100,
                    best_score_overall * 100,
                    best_score_iou_overall * 100,
                )
            )

    f_overall_iou = 0
    f_overall_dc = 0
    f_overall_sassd = 0
    f_overall_shd95 = 0
    for site_index in range(args.client_num):
        save_mode_path = os.path.join(model_path, "Overall_Site{}_best.pth".format(site_index + 1))
        net_clients[site_index].load_state_dict(torch.load(save_mode_path))
        args.load_path = model_path
        score_values = test_vis(
            site_index,
            args,
            dataloader_test_clients[site_index],
            net_clients[site_index],
            visulization=True,
        )
        print(
            "[Best] Site {} Dice: {:.2f} IOU: {:.2f} ASSD: {} HD95: {} ".format(
                site_index + 1,
                score_values[0] * 100,
                score_values[1] * 100,
                score_values[2],
                score_values[3],
            )
        )

        f_overall_dc += score_values[0]
        f_overall_iou += score_values[1]
        f_overall_sassd += score_values[2]
        f_overall_shd95 += score_values[3]

    f_overall_dc /= args.client_num
    f_overall_iou /= args.client_num
    f_overall_sassd /= args.client_num
    f_overall_shd95 /= args.client_num
    print("[Best] Dice Overall: {:.2f}  IOU Overall: {:.2f} ASSD: {} HD95: {} ".format(f_overall_dc * 100, f_overall_iou * 100, f_overall_sassd, f_overall_shd95))
