import argparse
import json
import numpy as np
import pickle
import random
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader, Subset
import torch.nn.functional as F
from models.base_model import BaseModel
from datasets import get_dataset
from utils.attackers import MiaAttack
import matplotlib.pyplot as plt
from utils.utils import roc_plot, get_new_fixed_dataset, print_set

parser = argparse.ArgumentParser(description='Membership inference Attacks')
parser.add_argument('device', default=0, type=int, help="GPU id to use")
parser.add_argument('config_path', default=0, type=str, help="config file path")
parser.add_argument('--dataset_name', default='mnist', type=str)
parser.add_argument('--model_name', default='mnist', type=str)
parser.add_argument('--model_num', default=256, type=int)
parser.add_argument('--num_cls', default=10, type=int)
parser.add_argument('--input_dim', default=1, type=int)
parser.add_argument('--image_size', default=28, type=int)
parser.add_argument('--seed', default=3407, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--attack_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--attacks', default='memia_gradnorm', type=str)
parser.add_argument('--attack_rapid', action='store_true', help="launch attacks")
parser.add_argument('--query_num', default=4, type=int, help="the times of multi-query")

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

    # Load Datasets
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
    with open(data_path, 'rb') as f:
        victim_train_list, victim_test_list, attack_train_list, attack_test_list, \
            tuning_train_list, tuning_test_list = pickle.load(f)
    print(f"Total Data Size: {total_size}")

    # Load Victim Model
    victim_train_dataset = Subset(total_dataset, victim_train_list)
    victim_test_dataset = Subset(total_dataset, victim_test_list)
    aug_victim_train_dataset = Subset(aug_total_dataset, victim_train_list)
    aug_victim_test_dataset = Subset(aug_total_dataset, victim_test_list)
    print(f"Victim Train Size: {len(victim_train_list)}, "
          f"Victim Test Size: {len(victim_test_list)}")
    victim_train_loader = DataLoader(victim_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                     pin_memory=False)
    victim_test_loader = DataLoader(victim_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                    pin_memory=False)
    aug_victim_train_loader = DataLoader(aug_victim_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                     pin_memory=False)
    aug_victim_test_loader = DataLoader(aug_victim_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                    pin_memory=False)

    victim_model_path = f"{base_folder}/victim_model/best.pth"
    print(f"Load Victim Model from {victim_model_path}")
    victim_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
    victim_model.load(victim_model_path, True)
    victim_model.test(aug_victim_train_loader, "Victim Model Train")
    victim_model.test(aug_victim_test_loader, "Victim Model Test")

    # Load Shadow Model
    shadow_train_dataset = Subset(total_dataset, attack_train_list)
    shadow_test_dataset = Subset(total_dataset, attack_test_list)
    aug_shadow_train_dataset = Subset(aug_total_dataset, attack_train_list)
    aug_shadow_test_dataset = Subset(aug_total_dataset, attack_test_list)
    print(f"Shadow Train Size: {len(attack_train_list)}, "
          f"Shadow Test Size: {len(attack_test_list)}")
    shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                     pin_memory=False)
    shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                    pin_memory=False)
    aug_shadow_train_loader = DataLoader(aug_shadow_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                     pin_memory=False)
    aug_shadow_test_loader = DataLoader(aug_shadow_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                    pin_memory=False)

    shadow_model_path = f"{base_folder}/shadow_model/best.pth"
    print(f"Load Shadow Model From {shadow_model_path}")
    shadow_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
    shadow_model.load(shadow_model_path, True)
    shadow_model.test(aug_shadow_train_loader, "Shadow Model Train")
    shadow_model.test(aug_shadow_test_loader, "Shadow Model Test")
        
    if args.attack_rapid:
        print(f"Rapid Train Size: {len(tuning_train_list)}")
        print(f"Rapid Test Size: {len(tuning_test_list)}")

        rapid_victim_in_model_list = []
        rapid_shadow_in_model_list = []
        rapid_victim_out_model_list = []
        rapid_shadow_out_model_list = []

        fixed_victim_test_loader_list = []
        fixed_victim_train_loader_list = []
        fixed_shadow_test_loader_list = []
        fixed_shadow_train_loader_list = []
        query = args.query_num
        
        for q_times in range(query):
            fixed_victim_train_loader_list.append(get_new_fixed_dataset(aug_victim_train_dataset, args.batch_size))
            fixed_victim_test_loader_list.append(get_new_fixed_dataset(aug_victim_test_dataset, args.batch_size))
            fixed_shadow_train_loader_list.append(get_new_fixed_dataset(aug_shadow_train_dataset, args.batch_size))
            fixed_shadow_test_loader_list.append(get_new_fixed_dataset(aug_shadow_test_dataset, args.batch_size))
        
        for idx in range(int(args.model_num)):
            rapid_victim_in_confidence_list = []
            rapid_victim_out_confidence_list = []
            
            victim_save_folder = f"{base_folder}/online/victim/model/{idx}"

            # Load rapid Victim
            rapid_victim_model_path = f"{victim_save_folder}/best.pth"
            print(f"Load online Victim Model from {rapid_victim_model_path}")
            rapid_victim_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
            #rapid_model(rapid_victim_model.model, amount, args.rapid_mode)
            rapid_victim_model.load(rapid_victim_model_path)
            
            for num in range(query):
                rapid_victim_in_confidences, rapid_victim_in_targets = rapid_victim_model.predict_target_loss(fixed_victim_train_loader_list[num])
                rapid_victim_out_confidences, rapid_victim_out_targets = rapid_victim_model.predict_target_loss(fixed_victim_test_loader_list[num])
                rapid_victim_in_confidence_list.append(rapid_victim_in_confidences)
                rapid_victim_out_confidence_list.append(rapid_victim_out_confidences)
            
            rapid_victim_in_model_list.append(torch.cat(rapid_victim_in_confidence_list, dim = 1).mean(dim=1, keepdim=True))
            rapid_victim_out_model_list.append(torch.cat(rapid_victim_out_confidence_list, dim = 1).mean(dim=1, keepdim=True))
        
        for idx in range(args.model_num):
            rapid_shadow_in_confidence_list = []
            rapid_shadow_out_confidence_list = []
        
            shadow_save_folder = f"{base_folder}/online/shadow/model/{idx}"

            # Load rapid Shadow
            rapid_shadow_model_path = f"{shadow_save_folder}/best.pth"
            print(f"Load online Shadow Model from {rapid_shadow_model_path}")
            rapid_shadow_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
            #rapid_model(rapid_shadow_model.model, amount, args.rapid_mode)
            rapid_shadow_model.load(rapid_shadow_model_path)
            
            
            for num in range(query):
                rapid_attack_in_confidences, rapid_attack_in_targets = rapid_shadow_model.predict_target_loss(aug_shadow_train_loader)
                rapid_attack_out_confidences, rapid_attack_out_targets = rapid_shadow_model.predict_target_loss(aug_shadow_test_loader)
                rapid_shadow_in_confidence_list.append(rapid_attack_in_confidences)
                rapid_shadow_out_confidence_list.append(rapid_attack_out_confidences)
            
            rapid_shadow_in_model_list.append(torch.cat(rapid_shadow_in_confidence_list, dim = 1).mean(dim=1, keepdim=True))
            rapid_shadow_out_model_list.append(torch.cat(rapid_shadow_out_confidence_list, dim = 1).mean(dim=1, keepdim=True))


        rapid_victim_in_confidences_tensor = torch.stack(rapid_victim_in_model_list, dim=1) 
        rapid_victim_out_confidences_tensor = torch.stack(rapid_victim_out_model_list, dim=1) 
        rapid_shadow_in_confidences_tensor = torch.stack(rapid_shadow_in_model_list, dim=1) 
        rapid_shadow_out_confidences_tensor = torch.stack(rapid_shadow_out_model_list, dim=1) 
        
##------------------------------------------------------------##       

        results_part1 = rapid_victim_in_confidences_tensor
        results_part2 = rapid_victim_out_confidences_tensor

        datalist = victim_train_list + victim_test_list
        point_path = f"{base_folder}/online/victim/point_to_sets.pkl"
        with open(point_path, 'rb') as f:
            point_to_sets = pickle.load(f)

        point_num = int(len(datalist)/2)
        dim_num = int(args.model_num/2)
        final_results_part2_a = torch.zeros(point_num, dim_num, 1)
        final_results_part1_a = torch.zeros(point_num, dim_num, 1)
        final_results_part1_b = torch.zeros(point_num, dim_num, 1)
        final_results_part2_b = torch.zeros(point_num, dim_num, 1)

        for i in range(point_num):
            models = point_to_sets[datalist[i]]
            all_models = set(range(int(args.model_num)))
            non_models = list(all_models - set(models))
            final_results_part1_a[i] = results_part1[i, models]
            final_results_part2_a[i] = results_part1[i, non_models]

        for i in range(point_num, int(point_num*2)):
            models = point_to_sets[datalist[i]]
            all_models = set(range(int(args.model_num)))
            non_models = list(all_models - set(models))
            final_results_part1_b[i - point_num] = results_part2[i - point_num, models]
            final_results_part2_b[i - point_num] = results_part2[i - point_num, non_models]

##------------------------------------------------------------##

        results_part3 = rapid_shadow_in_confidences_tensor
        results_part4 = rapid_shadow_out_confidences_tensor

        datalist = attack_train_list + attack_test_list
        point_path = f"{base_folder}/online/shadow/point_to_sets.pkl"
        with open(point_path, 'rb') as f:
            point_to_sets = pickle.load(f)

        final_results_part3_a = torch.zeros(point_num, dim_num, 1)
        final_results_part4_a = torch.zeros(point_num, dim_num, 1)
        final_results_part3_b = torch.zeros(point_num, dim_num, 1)
        final_results_part4_b = torch.zeros(point_num, dim_num, 1)
        
        
        for i in range(point_num):
            models = point_to_sets[datalist[i]]
            all_models = set(range(args.model_num))
            non_models = list(all_models - set(models))
            final_results_part3_a[i] = results_part3[i, models]
            final_results_part4_a[i] = results_part3[i, non_models]

        for i in range(point_num, int(point_num*2)):
            models = point_to_sets[datalist[i]]
            all_models = set(range(args.model_num))
            non_models = list(all_models - set(models))
            final_results_part3_b[i - point_num] = results_part4[i - point_num, models]
            final_results_part4_b[i - point_num] = results_part4[i - point_num, non_models]

        model_num = int(args.model_num/2)
        our_attackers = MiaAttack(
            victim_model, aug_victim_train_loader, aug_victim_test_loader,
            shadow_model, aug_shadow_train_loader, aug_shadow_test_loader,
            verify_victim_model_list=[[final_results_part2_a[:, :model_num].mean(dim=1, keepdim=True).squeeze(dim=1)], 
                                      [final_results_part2_b[:, :model_num].mean(dim=1, keepdim=True).squeeze(dim=1)]],
            verify_shadow_model_list=[[final_results_part4_a[:, :model_num].mean(dim=1, keepdim=True).squeeze(dim=1)], 
                                      [final_results_part4_b[:, :model_num].mean(dim=1, keepdim=True).squeeze(dim=1)]],
            device=device, num_cls=args.num_cls, epochs=args.attack_epochs, batch_size=args.batch_size,
            lr=0.0002, weight_decay=5e-4, optimizer="adam", scheduler="",
            dataset_name=args.dataset_name, model_name=args.model_name, query_num=args.query_num,
            attack_rapid=True
        )

    print("Start Membership Inference Attacks")
    
    pr_loss_tpr, pr_loss_auc, pr_loss_acc = our_attackers.rapid_attack()
    
    print(f"rapid attack results: tpr@0.1%fpr = {pr_loss_tpr*100:.2f}, auc = {pr_loss_auc:.3f}, accuracy = {pr_loss_acc:.3f}%")



if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config_path) as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)

    print_set(args)
    main(args)
