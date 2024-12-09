import argparse
import json
import numpy as np
import os
import pickle
import random
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from models.base_model import BaseModel
from datasets import get_dataset
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.utils import seed_worker, get_optimizer, get_scheduler, print_set, init_distributed, cleanup


parser = argparse.ArgumentParser()
parser.add_argument('config_path', default=0, type=str, help="config file path")
parser.add_argument('--device', default=0, type=int, help="GPU id to use")
parser.add_argument('--distributed', default=False, type=bool)
parser.add_argument('--world_size', default=1, type=int)
parser.add_argument('--dataset_name', default='mnist', type=str)
parser.add_argument('--model_name', default='mnist', type=str)
parser.add_argument('--num_cls', default=10, type=int)
parser.add_argument('--input_dim', default=3, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--model_num', default='2', type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--early_stop', default=0, type=int, help="patience for early stopping")
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--optimizer', default="sgd", type=str)
parser.add_argument('--scheduler', default="cosine", type=str)
            
def main_worker(rank, world_size, args):
    init_distributed(rank, world_size)
    print(f"Running on rank {rank}, total processes: {world_size}")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.distributed == False:
        device = f"cuda:{args.device}"
    else:
        device = f"cuda:{rank}"
        print(1)
    # cudnn.benchmark = True
    cudnn.deterministic = True
    cudnn.benchmark = False

    base_folder = f"results/{args.dataset_name}_{args.model_name}"
    if rank == 0:
        print(f"Base Folder: {base_folder}")

    trainset = get_dataset(args.dataset_name, train=True)
    testset = get_dataset(args.dataset_name, train=False)
    aug_trainset = get_dataset(args.dataset_name, train=True, augment=True)
    aug_testset = get_dataset(args.dataset_name, train=False, augment=True)
    if testset is None:
        total_dataset = trainset
        aug_total_dataset = aug_trainset
    else:
        total_dataset = ConcatDataset([trainset, testset])
        aug_total_dataset = ConcatDataset([aug_trainset, aug_testset])
    total_size = len(total_dataset)
    data_path = f"{base_folder}/data_index.pkl"
    if rank == 0:
        print(f"Total Data Size: {total_size}")

    with open(data_path, 'rb') as f:
        _, _, _, _, tuning_train_list, tuning_test_list = pickle.load(f)
    combined_list = tuning_train_list + tuning_test_list
    if rank == 0: 
        print(f"Train Size: {len(tuning_train_list)}, "
              f"Test Size: {len(tuning_test_list)}")

        victim_model_save_folder = f"{base_folder}/victim_model"

        if not os.path.exists(f"{victim_model_save_folder}/best.pth"):
            raise FileNotFoundError("no pretrained victim model")
        victim_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
        victim_model.load(f"{victim_model_save_folder}/best.pth")

        shadow_model_save_folder = f"{base_folder}/shadow_model"

        if not os.path.exists(f"{shadow_model_save_folder}/best.pth"):
            raise FileNotFoundError("no pretrained shadow model")
        shadow_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
        shadow_model.load(f"{shadow_model_save_folder}/best.pth")

    # Train reference model
    print(f"Train Reference Model")
    for idx in range(args.model_num):
        random.shuffle(combined_list)
        split_index = len(combined_list) // 2
        tuning_train_list = combined_list[:split_index]
        tuning_test_list = combined_list[split_index:2*split_index]
        victim_train_dataset = Subset(aug_total_dataset, tuning_train_list)
        victim_test_dataset = Subset(aug_total_dataset, tuning_test_list)
        
        if args.distributed:
            train_sampler = DistributedSampler(victim_train_dataset, num_replicas=world_size, rank=rank)
            victim_train_loader = DataLoader(victim_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                         pin_memory=True, sampler=train_sampler, worker_init_fn=seed_worker)
        else:
            victim_train_loader = DataLoader(victim_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                         pin_memory=True, worker_init_fn=seed_worker)
        if rank == 0:
            victim_test_loader = DataLoader(victim_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                        pin_memory=True, worker_init_fn=seed_worker)

        
        tuning_victim_model_save_folder = f"{base_folder}/RAPID/reference_model_{idx}"
        if not os.path.exists(tuning_victim_model_save_folder):
            os.makedirs(tuning_victim_model_save_folder, exist_ok=True)

        trans_victim_model = BaseModel(
                args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, 
                save_folder=tuning_victim_model_save_folder, device=device, 
                optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, 
                scheduler=args.scheduler, epochs=args.epochs)
        if args.distributed:
            trans_victim_model_model = DDP(trans_victim_model.model, device_ids=[rank])

            trans_victim_model.optimizer = get_optimizer(args.optimizer, 
                                                filter(lambda p: p.requires_grad, trans_victim_model_model.parameters()), 
                                                lr=args.lr, weight_decay=args.weight_decay)
        else:
            trans_victim_model.optimizer = get_optimizer(args.optimizer, 
                                                filter(lambda p: p.requires_grad, trans_victim_model.model.parameters()), 
                                                lr=args.lr, weight_decay=args.weight_decay)
        trans_victim_model.scheduler = get_scheduler(args.scheduler, 
                                            trans_victim_model.optimizer, args.epochs)
        
        best_acc = 0
        count = 0
        for epoch in range(args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            train_acc, train_loss = trans_victim_model.train(victim_train_loader, f"Epoch {epoch} Reference_{idx} Victim Train", args.distributed)
            if rank == 0:
                test_acc, test_loss = trans_victim_model.test(victim_test_loader, f"Epoch {epoch} Reference_{idx} Victim Test")
                if test_acc > best_acc:
                    best_acc = test_acc
                    trans_victim_model.save(epoch, test_acc, test_loss)
                    count = 0
                elif args.early_stop > 0:
                    count += 1
                    if count > args.early_stop:
                        print(f"Early Stop at Epoch {epoch}")
                        break
    cleanup()
if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config_path) as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
    print_set(args)
    if args.distributed:
        mp.spawn(main_worker, args=(args.world_size, args), nprocs=args.world_size)
    else:
        main_worker(0, 1, args)
