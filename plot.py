import argparse
import json
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from utils.utils import roc_plot
import matplotlib


parser = argparse.ArgumentParser(description='Membership inference Attacks ROC Plot')
parser.add_argument('device', default=0, type=int, help="GPU id to use")
parser.add_argument('config_path', default=0, type=str, help="config file path")
parser.add_argument('--dataset_name', default='cifar10', type=str)
parser.add_argument('--model_name', default='vgg16', type=str)
parser.add_argument('--num_cls', default=10, type=int)
parser.add_argument('--input_dim', default=3, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--attacks', default='', type=str)

def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = f"cuda:{args.device}"
    # cudnn.benchmark = True
    cudnn.deterministic = True
    cudnn.benchmark = False

    base_folder = f"results/{args.dataset_name}_{args.model_name}"
    print(f"Base Folder: {base_folder}")


    # Get attack type
    attacks = args.attacks.split(',')

    print("Start Membership Inference Attacks Plot")
    
    if "rapid_attack" in attacks:
        save_folder = f"results/{args.dataset_name}_{args.model_name}/RAPID/rapid_attack"
        our_acc, ROC_label, ROC_confidence_score = np.load(f'{save_folder}/Roc_confidence_score.npz')['acc'], np.load(f'{save_folder}/Roc_confidence_score.npz')['ROC_label'], np.load(f'{save_folder}/Roc_confidence_score.npz')['ROC_confidence_score']
        label= 'RAPID., acc=%.1f'%our_acc + '%'
        roc_plot(ROC_label, ROC_confidence_score, label, plot=True)

    matplotlib.rcParams.update({'font.size': 16})
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-3,1)
    plt.ylim(1e-3,1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(bbox_to_anchor=(0.7,0.75))
    plt.savefig(f'{save_folder}/roc_result.pdf', dpi=300, bbox_inches='tight')
    print(f"The curve has saved in {save_folder} folder")

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config_path) as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)

    main(args)
