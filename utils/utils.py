import numpy as np
import torch
import random
import os
from torch.nn import init
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def weight_init(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv') or classname == 'Linear':
        if getattr(m, 'bias', None) is not None:
            init.constant_(m.bias, 0.0)
        if getattr(m, 'weight', None) is not None:
            init.xavier_normal_(m.weight)
    elif 'Norm' in classname:
        if getattr(m, 'weight', None) is not None:
            m.weight.data.fill_(1)
        if getattr(m, 'bias', None) is not None:
            m.bias.data.zero_()


def get_model(model_type, num_cls, input_dim, num_submodule=2):
    if model_type == "resnet18":
        from models import ResNet18
        model = ResNet18(num_classes=num_cls)
    elif model_type == "resnet50":
        from models import ResNet50
        model = ResNet50(num_classes=num_cls)
    elif model_type == "vgg16":
        from models import VGG16
        model = VGG16(num_classes=num_cls)
    elif model_type == "densenet121":
        from models import DenseNet121
        model = DenseNet121(num_classes=num_cls)
    elif model_type == "mobilenetv2":
        from models import MobileNetV2
        model = MobileNetV2(num_classes=num_cls)
    elif model_type == "columnfc":
        from models import ColumnFC
        model = ColumnFC(input_dim=input_dim, output_dim=num_cls)
    elif model_type == "mia_fc":
        from models.attack_models import MIAFC
        model = MIAFC(input_dim=num_cls, output_dim=1)
    elif model_type == "mia_transformer":
        from models.attack_models import MIATransformer
        model = MIATransformer(input_dim=num_cls, output_dim=1)
    else:
        print(model_type)
        raise ValueError
    return model


def get_optimizer(optimizer_name, parameters, lr, weight_decay=0):
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    elif optimizer_name == "":
        optimizer = None
        # print("Do not use optimizer.")
    else:
        print(optimizer_name)
        raise ValueError
    return optimizer


def get_scheduler(scheduler_name, optimizer, epochs):
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs // 2, epochs * 3 // 4], gamma=0.1)
    elif scheduler_name == "":
        scheduler = None
        # print("Do not use scheduler.")
    else:
        print(scheduler_name)
        raise ValueError
    return scheduler


def set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def roc_plot(ROC_label, ROC_confidence_score, label='', plot=True):
    matplotlib.rcParams.update({'font.size': 16})
    ROC_confidence_score = np.nan_to_num(ROC_confidence_score,nan=np.nanmean(ROC_confidence_score))
    fpr, tpr, thresholds = roc_curve(ROC_label, ROC_confidence_score, pos_label=1)
    roc_auc = auc(fpr, tpr)
    if plot == True:
        low = tpr[np.where(fpr<.001)[0][-1]]
        plt.plot(fpr, tpr, label=label)
    else:
        return roc_auc

def get_new_fixed_dataset(aug_dataset, batch_size):
    inputs = [item[0] for item in aug_dataset]
    labels = [item[1] for item in aug_dataset]

    inputs_tensor = torch.stack(inputs)
    labels_tensor = torch.tensor(labels)

    fixed_dataset = TensorDataset(inputs_tensor, labels_tensor)
    fixed_dataloader = DataLoader(fixed_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                     pin_memory=False)
    return fixed_dataloader

def print_set(args):
    data = [(arg, value) for arg, value in vars(args).items()]

    max_arg_length = max(len(arg) for arg, _ in data)
    max_value_length = max(len(str(value)) for _, value in data)

    print("+-" + "-" * (max_arg_length + 1) + "+" + "-" * (max_value_length + 2) + "+")
    print(f"| { 'arguments'.ljust(max_arg_length) } | { 'values'.ljust(max_value_length) } |")
    print("+-" + "-" * (max_arg_length + 1) + "+" + "-" * (max_value_length + 2) + "+")
    for arg, value in data:
        print(f"| { arg.ljust(max_arg_length) } | { str(value).ljust(max_value_length) } |")
    print("+-" + "-" * (max_arg_length + 1) + "+" + "-" * (max_value_length + 2) + "+")

def init_distributed(rank, world_size, backend='nccl'):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12367' 
    init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up distributed training."""
    destroy_process_group()

def is_main_process():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True