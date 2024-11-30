import os
import shutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from models.base_model import BaseModel
from utils.utils import seed_worker
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils import roc_plot
import torch.distributions as dist
from sklearn.metrics import roc_curve, auc
import numpy as np
import heapq
import time
class MiaAttack:
    def __init__(self, victim_model, victim_train_loader, victim_test_loader,
                 shadow_model, shadow_train_loader, shadow_test_loader,
                 verify_victim_model_list=[], verify_shadow_model_list=[],
                 device="cuda", num_cls=10, lr=0.001, weight_decay=5e-4,
                 optimizer="adam", scheduler="", epochs=50, batch_size=128, 
                 dataset_name="mnist", model_name="mnist", query_num=1,
                 exploit_layer_list=[], attack_rapid=False):
        self.victim_model = victim_model
        self.victim_train_loader = victim_train_loader
        self.victim_test_loader = victim_test_loader
        self.shadow_model = shadow_model
        self.shadow_train_loader = shadow_train_loader
        self.shadow_test_loader = shadow_test_loader
        self.device = device
        self.num_cls = num_cls
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.query_num = query_num
        self.rapid_victim_model_list = verify_victim_model_list
        self.rapid_shadow_model_list = verify_shadow_model_list
        self._prepare_rapid()
    
    def _prepare_rapid(self):
        victim_in_confidence_list = []
        victim_out_confidence_list = []
        shadow_in_confidence_list = []
        shadow_out_confidence_list = []
        
        for i in tqdm(range(self.query_num)):
            victim_in_confidences, victim_in_targets = self.victim_model.predict_target_loss(self.victim_train_loader)
            victim_out_confidences, victim_out_targets = self.victim_model.predict_target_loss(self.victim_test_loader)
            victim_in_confidence_list.append(victim_in_confidences)
            victim_out_confidence_list.append(victim_out_confidences)

            attack_in_confidences, attack_in_targets = self.shadow_model.predict_target_loss(self.shadow_train_loader)
            attack_out_confidences, attack_out_targets = self.shadow_model.predict_target_loss(self.shadow_test_loader)
            shadow_in_confidence_list.append(attack_in_confidences)
            shadow_out_confidence_list.append(attack_out_confidences)
            
        self.attack_in_confidences, self.attack_in_targets = \
            torch.cat(shadow_in_confidence_list, dim = 1).mean(dim = 1, keepdim = True), attack_in_targets
        self.attack_out_confidences, self.attack_out_targetss = \
            torch.cat(shadow_out_confidence_list, dim = 1).mean(dim = 1, keepdim = True), attack_out_targets
        
        self.victim_in_confidences, self.victim_in_targets = \
            torch.cat(victim_in_confidence_list, dim = 1).mean(dim = 1, keepdim = True), victim_in_targets
        self.victim_out_confidences, self.victim_out_targets = \
            torch.cat(victim_out_confidence_list, dim = 1).mean(dim = 1, keepdim = True), victim_out_targets
            
        self.rapid_attack_in_confidences = torch.cat(self.rapid_shadow_model_list[0], dim=1)
        self.rapid_attack_out_confidences = torch.cat(self.rapid_shadow_model_list[1], dim=1)
        self.rapid_victim_in_confidences = torch.cat(self.rapid_victim_model_list[0], dim=1)
        self.rapid_victim_out_confidences = torch.cat(self.rapid_victim_model_list[1], dim=1)

    def rapid_attack(self, model_name="mia_fc"):
        
        attack_losses = torch.cat([self.attack_in_confidences, self.attack_out_confidences], dim=0)
        rapid_attack_losses = torch.cat([self.rapid_attack_in_confidences, self.rapid_attack_out_confidences], dim=0)
        attack_labels = torch.cat([torch.ones(self.attack_in_confidences.size(0)),
                                   torch.zeros(self.attack_out_confidences.size(0))], dim=0).unsqueeze(1)

        victim_losses = torch.cat([self.victim_in_confidences, self.victim_out_confidences], dim=0)
        rapid_victim_losses = torch.cat([self.rapid_victim_in_confidences, self.rapid_victim_out_confidences], dim=0)
        victim_labels = torch.cat([torch.ones(self.victim_in_confidences.size(0)),
                                   torch.zeros(self.victim_out_confidences.size(0))], dim=0).unsqueeze(1)
        
        save_folder = f"results/{self.dataset_name}_{self.model_name}/RAPID/rapid_attack"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        new_attack_data = torch.cat([attack_losses, attack_losses - rapid_attack_losses], dim=1)
        new_victim_data = torch.cat([victim_losses, victim_losses - rapid_victim_losses], dim=1)
        #new_attack_data = torch.cat([attack_losses], dim=1)
        #new_victim_data = torch.cat([victim_losses], dim=1)
        attack_model_save_folder = save_folder
        
        attack_train_dataset = TensorDataset(new_attack_data, attack_labels)
        attack_test_dataset = TensorDataset(new_victim_data, victim_labels)
        
        attack_train_dataloader = DataLoader(
            attack_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True,
            worker_init_fn=seed_worker)
        attack_test_dataloader = DataLoader(
            attack_test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True,
            worker_init_fn=seed_worker)

        # no assemble
        
        attack_model = BaseModel(
            model_name, device=self.device, save_folder=attack_model_save_folder, num_cls=new_victim_data.size(1), 
            optimizer=self.optimizer, lr=self.lr, weight_decay=self.weight_decay, scheduler=self.scheduler, epochs=self.epochs)

        best_acc = 0
        best_tpr = 0
        for epoch in range(self.epochs):
            train_acc, train_loss = attack_model.attack_train(attack_train_dataloader)
            test_acc, test_loss = attack_model.attack_test(attack_test_dataloader)
            test_acc_plot, test_loss_plot, ROC_label, ROC_confidence_score = attack_model.plot_test(attack_test_dataloader)
            ROC_confidence_score = np.nan_to_num(ROC_confidence_score,nan=np.nanmean(ROC_confidence_score))
            fpr, tpr, thresholds = roc_curve(ROC_label, ROC_confidence_score, pos_label=1)
            low = tpr[np.where(fpr<.001)[0][-1]]
            if low > best_tpr:
                best_tpr = low
                best_acc = test_acc
                attack_model.save(epoch, test_acc, test_loss)
                #print(type(ROC_label),type(ROC_confidence_score),type(best_acc))
                np.savez(f'{save_folder}Roc_confidence_score.npz',acc=best_acc.cpu(),ROC_label=ROC_label,ROC_confidence_score=ROC_confidence_score)

            best_auc = roc_plot(ROC_label, ROC_confidence_score, plot=False)
        return best_tpr, best_auc, best_acc
        
        
class MiaAttackPlot:
    def __init__(self, victim_model, victim_train_loader, victim_test_loader,
                 verify_victim_model_list=None, device="cuda", num_cls=10, batch_size=128, 
                 dataset_name="mnist", model_name="mnist", 
                 exploit_layer_list=[], attack_original=False, 
                 attack_reference=False, attack_rapid=False,
                 attack_trajectory=False, attack_lira=False,
                 attack_distill=False):
        self.victim_model = victim_model
        self.victim_train_loader = victim_train_loader
        self.victim_test_loader = victim_test_loader
        self.device = device
        self.num_cls = num_cls
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.model_name = model_name
        if attack_original:
            self._prepare_orig()
        elif attack_rapid:
            self.rapid_victim_model_list = verify_victim_model_list
            self._prepare_rapid()

            
    def _prepare_orig(self):
        '''
        self.attack_in_predicts, self.attack_in_targets, self.attack_in_losses, self.attack_in_sens = \
            self.shadow_model.predict_target_sensitivity(self.shadow_train_loader)
        self.attack_out_predicts, self.attack_out_targets, self.attack_out_losses, self.attack_out_sens = \
            self.shadow_model.predict_target_sensitivity(self.shadow_test_loader)
        '''
        self.victim_in_predicts, self.victim_in_targets, self.victim_in_losses, self.victim_in_sens = \
            self.victim_model.predict_target_sensitivity(self.victim_train_loader)
        self.victim_out_predicts, self.victim_out_targets, self.victim_out_losses, self.victim_out_sens = \
            self.victim_model.predict_target_sensitivity(self.victim_test_loader)
        
    def _prepare_rapid(self):
        victim_in_confidence_list = []
        victim_out_confidence_list = []

        for i in range(8):
            victim_in_confidences, victim_in_targets = self.victim_model.predict_target_loss(self.victim_train_loader)
            victim_out_confidences, victim_out_targets = self.victim_model.predict_target_loss(self.victim_test_loader)
            victim_in_confidence_list.append(victim_in_confidences)
            victim_out_confidence_list.append(victim_out_confidences)
        
        self.victim_in_confidences, self.victim_in_targets = \
            torch.cat(victim_in_confidence_list, dim = 1).mean(dim = 1, keepdim = True), victim_in_targets
        self.victim_out_confidences, self.victim_out_targets = \
            torch.cat(victim_out_confidence_list, dim = 1).mean(dim = 1, keepdim = True), victim_out_targets
            
        self.rapid_victim_in_confidences = torch.cat(self.rapid_victim_model_list[0], dim=1)
        self.rapid_victim_out_confidences = torch.cat(self.rapid_victim_model_list[1], dim=1)

    def nn_attack(self, mia_type="nn", model_name="mia_fc"):
        victim_predicts = torch.cat([self.victim_in_predicts, self.victim_out_predicts], dim=0)
        victim_sens = torch.cat([self.victim_in_sens, self.victim_out_sens], dim=0)
        victim_targets = torch.cat([self.victim_in_targets, self.victim_out_targets], dim=0)
        victim_targets = F.one_hot(victim_targets, num_classes=self.num_cls).float()
        victim_losses = torch.cat([self.victim_in_losses, self.victim_out_losses], dim=0)
        victim_labels = torch.cat([torch.ones(self.victim_in_targets.size(0)),
                                   torch.zeros(self.victim_out_targets.size(0))], dim=0).unsqueeze(1)

        save_folder = f"results/{self.dataset_name}_{self.model_name}/blackbox"
        if not os.path.exists(save_folder):
            raise FileNotFoundError("no pretrained attack models")

        if mia_type == "nn_cls":
            new_victim_data = torch.cat([victim_predicts, victim_targets], dim=1)
            attack_model_save_folder = save_folder + "/nn_cls"
        elif mia_type == "nn_top3":
            new_victim_data, _ = torch.topk(victim_predicts, k=3, dim=-1)
            attack_model_save_folder = save_folder + "/nn_top3"
        elif mia_type == "nn_sens_cls":
            new_victim_data = torch.cat([victim_predicts, victim_sens, victim_targets], dim=1)
            attack_model_save_folder = save_folder + "/samia"
        elif mia_type == "nn_loss":
            new_victim_data = victim_losses
            attack_model_save_folder = save_folder + "/nn_loss"
        else:
            new_victim_data = victim_predicts
            attack_model_save_folder = save_folder + "/nn"

        attack_test_dataset = TensorDataset(new_victim_data, victim_labels)
        attack_test_dataloader = DataLoader(
            attack_test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True,
            worker_init_fn=seed_worker)

        if not os.path.exists(attack_model_save_folder):
            raise FileNotFoundError(f"no pretrained attack model save folder: {attack_model_save_folder}")
        attack_model = BaseModel(
            model_name, device=self.device, num_cls=new_victim_data.size(1))
        attack_model.load(f"{attack_model_save_folder}/best.pth")
        test_acc, test_loss, ROC_label, ROC_confidence_score = attack_model.plot_test(attack_test_dataloader)

        return test_acc, ROC_label, ROC_confidence_score
    
    
    def rapid_attack(self, model_name="mia_fc"):
        
        victim_losses = torch.cat([self.victim_in_confidences, self.victim_out_confidences], dim=0)
        rapid_victim_losses = torch.cat([self.rapid_victim_in_confidences, self.rapid_victim_out_confidences], dim=0)
        victim_labels = torch.cat([torch.ones(self.victim_in_confidences.size(0)),
                                   torch.zeros(self.victim_out_confidences.size(0))], dim=0).unsqueeze(1)

        save_folder = f"results/{self.dataset_name}_{self.model_name}/ours/rapid_lira_attack"
        if not os.path.exists(save_folder):
            raise FileNotFoundError("no pretrained attack models")

        new_victim_data = torch.cat([victim_losses, rapid_victim_losses], dim=1)
        attack_model_save_folder = save_folder
    
        attack_test_dataset = TensorDataset(new_victim_data, victim_labels)
        attack_test_dataloader = DataLoader(
            attack_test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True,
            worker_init_fn=seed_worker)


        attack_model = BaseModel(model_name, device=self.device, num_cls=new_victim_data.size(1))
        attack_model.load(f"{attack_model_save_folder}/best.pth")

        test_acc, test_loss, ROC_label, ROC_confidence_score = attack_model.plot_test(attack_test_dataloader)

        return test_acc, ROC_label, ROC_confidence_score
    
    