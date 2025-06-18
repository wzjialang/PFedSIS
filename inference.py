"""
Visualization module for PFedSIS federated learning experiments.

This module provides functionality to visualize and test federated learning models
on medical/robotics datasets using various model architectures.
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from scripts.tester_utils_med import test_vis
from net.Ours.pfedsis import BuildFPN as our_BuildFPN
import torchvision.transforms as transforms
from dataloaders.transforms import *

# Constants
ROBOTICS_CLIENT_NUM = 3
ROBOTICS_NUM_CLASSES = 4
ROBOTICS_CHANNELS = 3
DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_WORKERS = 3

# Argument parsing
parser = argparse.ArgumentParser(description="PFedSIS evaluation/visualization")
parser.add_argument(
    "--load_path",
    type=str,
    default="./weight",
    help="Path to load model weight",
)
parser.add_argument("--dataset", type=str, default="robotics", help="Dataset name")
parser.add_argument("--gpu", type=str, default="cuda:0", help="GPU to use")
args = parser.parse_args()

print(f"Arguments: {args}")

# Dataset configuration
if args.dataset == "robotics":
    from dataloaders.robotics_dataloader import Dataset

    args.client_num = ROBOTICS_CLIENT_NUM
    args.num_classes = ROBOTICS_NUM_CLASSES
    args.c_in = ROBOTICS_CHANNELS
else:
    raise NotImplementedError(f"Dataset '{args.dataset}' is not implemented")

print(f"Updated arguments: {args}")


if __name__ == "__main__":
    # Initialize client storage lists
    dataloader_test_clients = []
    net_ours_clients = []
    test_transforms = transforms.Compose([ToTensor()])
    # Setup clients
    for client_idx in range(args.client_num):
        # Construct model path
        model_path = os.path.join(args.load_path, f"Overall_Site{client_idx + 1}_best.pth")

        # Create test dataset
        dataset_test = Dataset(client_idx=client_idx, split="test", transform=test_transforms)

        # Create dataloader
        dataloader_test = DataLoader(
            dataset_test,
            batch_size=DEFAULT_BATCH_SIZE,
            shuffle=False,
            num_workers=DEFAULT_NUM_WORKERS,
            pin_memory=True,
        )
        dataloader_test_clients.append(dataloader_test)

        # Initialize and load model
        net_ours = our_BuildFPN(args.num_classes, "pvtb0", "fpn").to(args.gpu)
        net_ours.load_state_dict(torch.load(model_path), strict=False)
        net_ours_clients.append(net_ours)

    # Evaluation phase
    print("Starting evaluation...")
    for site_index in range(len(net_ours_clients)):
        ours_net = net_ours_clients[site_index]
        score_values = test_vis(
            site_index,
            args,
            dataloader_test_clients[site_index],
            ours_net,
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

        f_overall_dc /= args.client_num
        f_overall_iou /= args.client_num
        f_overall_sassd /= args.client_num
        f_overall_shd95 /= args.client_num

    print("[Best] Dice Overall: {:.2f} IOU Overall: {:.2f} ASSD: {} HD95: {} ".format(f_overall_dc * 100, f_overall_iou * 100, f_overall_shd95, f_overall_sassd))
    print("Evaluation completed.")
