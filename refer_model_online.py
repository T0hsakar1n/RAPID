#70 71
import argparse
import json
import numpy as np
import os
import pickle
import random
import copy
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from models.base_model import BaseModel
from datasets import get_dataset
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.utils import seed_worker, get_optimizer, get_scheduler, set_module, print_set, init_distributed, cleanup
from collections import Counter, defaultdict
parser = argparse.ArgumentParser()
parser.add_argument('config_path', default=0, type=str, help="config file path")
parser.add_argument('--device', default=0, type=int, help="GPU id to use")
parser.add_argument('--distributed', default=False, type=bool)
parser.add_argument('--world_size', default=1, type=int)
parser.add_argument('--dataset_name', default='mnist', type=str)
parser.add_argument('--model_name', default='mnist', type=str)
parser.add_argument('--model_num', default=256, type=int)
parser.add_argument('--num_cls', default=10, type=int)
parser.add_argument('--input_dim', default=1, type=int)
parser.add_argument('--seed', default=667, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--early_stop', default=0, type=int, help="patience for early stopping")
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--optimizer', default="sgd", type=str)
parser.add_argument('--scheduler', default="cosine", type=str)
parser.add_argument('--state', default="victim", type=str, help="definite which type refer model")

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
        victim_train_list, victim_test_list, attack_train_list, attack_test_list, tuning_train_list, tuning_test_list = pickle.load(f)

    #combined_list = attack_train_list + attack_test_list + tuning_train_list + tuning_test_list
    if args.state == 'victim':
        train_list = victim_train_list + victim_test_list
    elif args.state == 'shadow':
        train_list = attack_train_list + attack_test_list

    if len(train_list) == len(set(train_list)):
        print("All elements in train_list are unique.")
    else:
        print("There are duplicate elements in train_list.")
        exit()

    total_points = len(train_list)
    points_per_set = int(total_points/2)
    sets_needed = args.model_num
    max_occurrences = int(sets_needed/2)
    point_selection_counter = Counter({point: 0 for point in train_list})
    sets = []
    
    save_folder = f"results/{args.dataset_name}_{args.model_name}/online/{args.state}"
    print(f"Save Folder: {save_folder}")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    output_path = f"{save_folder}/sampled_data_sets.pkl"
    if not os.path.exists(output_path):
        print(f"don't find the sets file in {save_folder}, create one")
        for set_index in range(0, sets_needed):
            candidates = [point for point, count in point_selection_counter.items() if count < max_occurrences]
            if len(candidates) < points_per_set:
                raise ValueError(f"Not enough points left to create set {set_index + 1}. Only {len(candidates)} points remaining.")

            random.shuffle(candidates)
            sorted_candidates = sorted(candidates, key=lambda x: point_selection_counter[x])
            current_set = sorted_candidates[:points_per_set]
            sets.append(current_set)
            point_selection_counter.update(current_set)

        for point in train_list:
            assert point_selection_counter[point] == max_occurrences, f"Point {point} was selected {point_selection_counter[point]} times, expected {max_occurrences} times."
        
        with open(output_path, 'wb') as f:
            pickle.dump(sets, f)

        point_to_sets_path = f"{save_folder}/point_to_sets.pkl"
        point_to_sets = defaultdict(list)
        for set_index, point_set in enumerate(sets):
            for point in point_set:
                point_to_sets[point].append(set_index)

        with open(point_to_sets_path, 'wb') as f:
            pickle.dump(point_to_sets, f)
    else:
        with open(output_path, 'rb') as f:
            sets = pickle.load(f)

    print("start training")
    for idx in range(0, sets_needed):
        random.shuffle(sets[idx])
        victim_train_list = sets[idx]
        victim_test_list = list((Counter(train_list)-Counter(sets[idx])).elements())
        # print(f"Reference Train Size: {len(victim_train_list)}, "
        #     f"Reference Test Size: {len(victim_test_list)}")
        victim_train_dataset = Subset(aug_total_dataset, victim_train_list)
        victim_test_dataset = Subset(aug_total_dataset, victim_test_list)
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
        model_save_folder = f"{save_folder}/model/{idx}"
        print(f"Train Model")
        if not os.path.exists(model_save_folder):
            os.makedirs(model_save_folder)
        
        reference_model = BaseModel(
            args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, 
            save_folder=model_save_folder, device=device, 
            optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, 
            scheduler=args.scheduler, epochs=args.epochs)
        
        if args.distributed:
            reference_model_model = DDP(reference_model.model, device_ids=[rank])

            reference_model.optimizer = get_optimizer(args.optimizer, 
                                                filter(lambda p: p.requires_grad, reference_model_model.parameters()), 
                                                lr=args.lr, weight_decay=args.weight_decay)
        else:
            reference_model.optimizer = get_optimizer(args.optimizer, 
                                                filter(lambda p: p.requires_grad, reference_model.model.parameters()), 
                                                lr=args.lr, weight_decay=args.weight_decay)
        reference_model.scheduler = get_scheduler(args.scheduler, 
                                            reference_model.optimizer, args.epochs)
        best_acc = 0
        count = 0
        for epoch in range(args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            train_acc, train_loss = reference_model.train(victim_train_loader, f"Epoch {epoch} Shadow Train", args.distributed)
            if rank == 0:
                test_acc, test_loss = reference_model.test(victim_test_loader, f"Epoch {epoch} Shadow Test")
                if test_acc > best_acc:
                    best_acc = test_acc
                    reference_model.save(epoch, test_acc, test_loss)
                    count = 0
                elif args.early_stop > 0:
                    count += 1
                    if count > args.early_stop:
                        print(f"Early Stop at Epoch {epoch}")
                        break
        del reference_model
        del victim_train_loader, victim_test_loader
        del victim_train_dataset, victim_test_dataset
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
