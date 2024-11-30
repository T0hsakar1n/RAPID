import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from utils.utils import get_model, get_optimizer, get_scheduler, weight_init, is_main_process
import torch.distributed as dist
from tqdm import tqdm
class BaseModel:
    def __init__(self, model_type, device="cuda", save_folder="", num_cls=10, input_dim=100, num_sub=2,
                 optimizer="", lr=0, weight_decay=0, scheduler="", epochs=0, attack_model_type=""):
        self.model = get_model(model_type, num_cls, input_dim, num_submodule=num_sub)
        self.model.to(device)
        self.model.apply(weight_init)
        self.device = device
        if epochs == 0:
            self.optimizer = None
            self.scheduler = None
        else:
            self.optimizer = get_optimizer(optimizer, self.model.parameters(), lr, weight_decay)
            self.scheduler = get_scheduler(scheduler, self.optimizer, epochs)
        if model_type in ["mia_fc", "mia_transformer", "mia_me"]:
            self.criterion = nn.BCEWithLogitsLoss()
            #self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.save_pref = save_folder
        self.num_cls = num_cls

        if attack_model_type:
            self.attack_model = get_model(attack_model_type, num_cls*2, 2)
            self.attack_model.to(device)
            self.attack_model.apply(weight_init)
            self.attack_model_optim = get_optimizer("adam", self.attack_model.parameters(), lr=0.001, weight_decay=5e-4)

    def train(self, train_loader, log_pref="", distributed=False):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

        if distributed:
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            correct_tensor = torch.tensor(correct, device=self.device)
            total_tensor = torch.tensor(total, device=self.device)

            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        
            total_loss = total_loss_tensor.item()
            correct = correct_tensor.item()
            total = total_tensor.item()
        
        if self.scheduler:
            self.scheduler.step()
        acc = 100. * correct / total
        total_loss /= total
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss
    
    def attack_train(self, train_loader, log_pref=""):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            predicted = torch.round(torch.sigmoid(outputs))
            correct += predicted.eq(targets).sum().item()
        if self.scheduler:
            self.scheduler.step()
        acc = 100. * correct / total
        total_loss /= total
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss
    '''
    # attack_train v2
    def attack_train(self, train_loader, log_pref=""):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets = targets.round().long()
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            outputs = F.softmax(outputs, dim=-1)
            loss = self.criterion(outputs, targets.view(-1))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
        if self.scheduler:
            self.scheduler.step()
        acc = 100. * correct / total
        total_loss /= total
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss
    # attack train v2
    '''
    def me_attack_train(self, train_loader, log_pref=""):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        for inputs, trans_inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            trans_inputs = trans_inputs.to(self.device) 
            targets = targets.to(self.device)
            inputs_list = [inputs, trans_inputs]
            self.optimizer.zero_grad()
            outputs = self.model(inputs_list)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            predicted = torch.round(torch.sigmoid(outputs))
            correct += predicted.eq(targets).sum().item()
        if self.scheduler:
            self.scheduler.step()
        acc = 100. * correct / total
        total_loss /= total
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss

    def me_split_attack_train(self, train_loader, log_pref=""):
        self.model.extractors.eval()
        self.model.classifier.train()
        total_loss = 0
        correct = 0
        total = 0
        for inputs, trans_inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            trans_inputs = trans_inputs.to(self.device) 
            targets = targets.to(self.device)
            inputs_list = [inputs, trans_inputs]
            self.optimizer.zero_grad()
            outputs = self.model(inputs_list)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            predicted = torch.round(torch.sigmoid(outputs))
            correct += predicted.eq(targets).sum().item()
        if self.scheduler:
            self.scheduler.step()
        acc = 100. * correct / total
        total_loss /= total
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss

    def train_defend_ppb(self, train_loader, defend_arg=None, log_pref=""):
        self.model.train()
        total_loss = 0
        total_loss1 = 0
        total_loss2 = 0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            self.optimizer.zero_grad()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss1 = self.criterion(outputs, targets)
            ranked_outputs, _ = torch.topk(outputs, self.num_cls, dim=-1)
            size = targets.size(0)
            even_size = size // 2 * 2
            if even_size > 0:
                loss2 = F.kl_div(F.log_softmax(ranked_outputs[:even_size // 2], dim=-1),
                                 F.softmax(ranked_outputs[even_size // 2:even_size], dim=-1),
                                 reduction='batchmean')
            else:
                loss2 = torch.zeros(1).to(self.device)
            loss = loss1 + defend_arg * loss2
            total_loss += loss.item() * size
            total_loss1 += loss1.item() * size
            total_loss2 += loss2.item() * size
            total += size
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            loss.backward()
            self.optimizer.step()
        acc = 100. * correct / total
        total_loss /= total
        total_loss1 /= total
        total_loss2 /= total

        if self.scheduler:
            self.scheduler.step()
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}, Loss1 {:.3f}, Loss2 {:.3f}".format(
                log_pref, acc, total_loss, total_loss1, total_loss2))
        return acc, total_loss

    def train_defend_adv(self, train_loader, test_loader, privacy_theta=1.0, log_pref=""):
        """
        modified from
        https://github.com/Lab41/cyphercat/blob/master/Defenses/Adversarial_Regularization.ipynb
        """
        total_loss = 0
        correct = 0
        total = 0
        infer_iterations = 7
        # train adversarial network

        train_iter = iter(train_loader)
        test_iter = iter(test_loader)
        train_iter2 = iter(train_loader)

        self.model.eval()
        self.attack_model.train()
        for infer_iter in range(infer_iterations):
            with torch.no_grad():
                try:
                    inputs, targets = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    inputs, targets = next(train_iter)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                in_predicts = F.softmax(self.model(inputs), dim=-1)
                in_targets = F.one_hot(targets, num_classes=self.num_cls).float()

                try:
                    inputs, targets = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_loader)
                    inputs, targets = next(test_iter)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                out_predicts = F.softmax(self.model(inputs), dim=-1)
                out_targets = F.one_hot(targets, num_classes=self.num_cls).float()

                infer_train_data = torch.cat([torch.cat([in_predicts, in_targets], dim=-1),
                                              torch.cat([out_predicts, out_targets], dim=-1)], dim=0)
                infer_train_label = torch.cat([torch.ones(in_predicts.size(0)),
                                               torch.zeros(out_predicts.size(0))]).long().to(self.device)

            self.attack_model_optim.zero_grad()
            infer_loss = privacy_theta * F.cross_entropy(self.attack_model(infer_train_data), infer_train_label)
            infer_loss.backward()
            self.attack_model_optim.step()

        self.model.train()
        self.attack_model.eval()
        try:
            inputs, targets = next(train_iter2)
        except StopIteration:
            train_iter2 = iter(train_loader)
            inputs, targets = next(train_iter2)
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss1 = self.criterion(outputs, targets)
        in_predicts = F.softmax(outputs, dim=-1)
        in_targets = F.one_hot(targets, num_classes=self.num_cls).float()
        infer_data = torch.cat([in_predicts, in_targets], dim=-1)
        infer_labels = torch.ones(targets.size(0)).long().to(self.device)
        infer_loss = F.cross_entropy(self.attack_model(infer_data), infer_labels)
        loss = loss1 - privacy_theta * infer_loss
        loss.backward()
        self.optimizer.step()
        total_loss += loss.item() * targets.size(0)
        total += targets.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        if self.scheduler:
            self.scheduler.step()
        acc = 100. * correct / total
        total_loss /= total
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss

    def test(self, test_loader, log_pref=""):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

        acc = 100. * correct / total
        total_loss /= total
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss
    
    def attack_test(self, test_loader, log_pref=""):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)
                correct += torch.sum(torch.round(torch.sigmoid(outputs)) == targets)
                total += targets.size(0)

        acc = 100. * correct / total
        total_loss /= total
        if log_pref:
            print("{}: Accuracy {:.3f}%, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss
    '''
    # attack_test v2
    def attack_test(self, test_loader, log_pref=""):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = targets.round().long()
                outputs = self.model(inputs)
                outputs = F.softmax(outputs, dim=-1)
                loss = self.criterion(outputs, targets.view(-1))
                total_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

        acc = 100. * correct / total
        total_loss /= total
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss
    # attack_test v2
    '''
    def me_attack_test(self, test_loader, log_pref=""):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, trans_inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(self.device)
                trans_inputs = trans_inputs.to(self.device)
                targets = targets.to(self.device)
                new_inputs = [inputs, trans_inputs]
                outputs = self.model(new_inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)
                correct += torch.sum(torch.round(torch.sigmoid(outputs)) == targets)
                total += targets.size(0)

        acc = 100. * correct / total
        total_loss /= total
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss
    
    def plot_test(self, test_loader, log_pref=""):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            ROC_confidence_score = torch.empty(0).to(self.device)
            ROC_label = torch.empty(0).to(self.device)
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)
                correct += torch.sum(torch.round(torch.sigmoid(outputs)) == targets)
                total += targets.size(0)

                ROC_confidence_score = torch.cat((ROC_confidence_score, torch.sigmoid(outputs)))
                ROC_label = torch.cat((ROC_label, targets))

        acc = 100. * correct / total
        total_loss /= total
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss, torch.squeeze(ROC_label).cpu().numpy(), torch.squeeze(ROC_confidence_score).cpu().numpy()
    '''
    # plot_test v2
    def plot_test(self, test_loader, log_pref=""):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            ROC_confidence_score = torch.empty(0).to(self.device)
            ROC_label = torch.empty(0).to(self.device)
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = targets.round().long()
                outputs = self.model(inputs)
                outputs = F.softmax(outputs, dim=-1)
                loss = self.criterion(outputs, targets.view(-1))
                total_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
                ROC_confidence_score = torch.cat((ROC_confidence_score, outputs[:, 1].unsqueeze(1)))
                ROC_label = torch.cat((ROC_label, targets))

        acc = 100. * correct / total
        total_loss /= total
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss, torch.squeeze(ROC_label).cpu().numpy(), torch.squeeze(ROC_confidence_score).cpu().numpy()
    # plot_test v2
    '''
    def me_plot_test(self, test_loader, log_pref=""):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            ROC_confidence_score = torch.empty(0).to(self.device)
            ROC_label = torch.empty(0).to(self.device)
            for batch_idx, (inputs, trans_inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(self.device)
                trans_inputs = trans_inputs.to(self.device)
                targets = targets.to(self.device)
                new_inputs = [inputs, trans_inputs]
                outputs = self.model(new_inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)
                correct += torch.sum(torch.round(torch.sigmoid(outputs)) == targets)
                total += targets.size(0)

                ROC_confidence_score = torch.cat((ROC_confidence_score, torch.sigmoid(outputs)))
                ROC_label = torch.cat((ROC_label, targets))

        acc = 100. * correct / total
        total_loss /= total
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss, torch.squeeze(ROC_label).cpu().numpy(), torch.squeeze(ROC_confidence_score).cpu().numpy()

    def save(self, epoch, acc, loss):
        # save_path = f"{self.save_pref}/{epoch}.pth"
        save_path = f"{self.save_pref}/best.pth"
        state = {
            'epoch': epoch + 1,
            'acc': acc,
            'loss': loss,
            'state': self.model.state_dict()
        }
        torch.save(state, save_path)
        return save_path
    
    def distill_save(self, epoch, acc, loss):
        save_path = f"{self.save_pref}/{epoch}.pth"
        # save_path = f"{self.save_pref}/best.pth"
        state = {
            'epoch': epoch + 1,
            'acc': acc,
            'loss': loss,
            'state': self.model.state_dict()
        }
        torch.save(state, save_path)
        return save_path

    def load(self, load_path, verbose=False):
        state = torch.load(load_path, map_location=self.device)
        acc = state['acc']
        if verbose:
            # print(f"Load model from {load_path}")
            print(f"Epoch {state['epoch']}, Acc: {state['acc']:.3f}, Loss: {state['loss']:.3f}")
        self.model.load_state_dict(state['state'])
        return acc

    def predict_target_sensitivity(self, data_loader, m=10, epsilon=1e-3):
        self.model.eval()
        predict_list = []
        target_list = []
        loss_list = []
        sensitivity_list = []
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                predicts = F.softmax(outputs, dim=-1)
                log_predicts = torch.log(predicts)
                losses = F.nll_loss(log_predicts, targets, reduction='none')
                losses = torch.unsqueeze(losses, 1)

                predict_list.append(predicts.detach().data.cpu())
                target_list.append(targets.detach().data.cpu())
                loss_list.append(losses.detach().data.cpu())

                if len(inputs.size()) == 4:
                    x = inputs.repeat((m, 1, 1, 1))
                elif len(inputs.size()) == 3:
                    x = inputs.repeat((m, 1, 1))
                elif len(inputs.size()) == 2:
                    x = inputs.repeat((m, 1))
                u = torch.randn_like(x)
                evaluation_points = x + epsilon * u
                new_predicts = F.softmax(self.model(evaluation_points), dim=-1)
                diff = torch.abs(new_predicts - predicts.repeat((m, 1)))
                diff = diff.view(m, -1, self.num_cls)
                sensitivity = diff.mean(dim=0) / epsilon
                sensitivity_list.append(sensitivity.detach().data.cpu())

        predicts = torch.cat(predict_list, dim=0)
        targets = torch.cat(target_list, dim=0)
        losses = torch.cat(loss_list, dim=0)
        sensitivities = torch.cat(sensitivity_list, dim=0)
        return predicts, targets, losses, sensitivities

    def predict_target_gradnorm(self, data_loader, layer_list):
        self.model.eval()
        for name, param in self.model.named_parameters():
            name_prefix = name.split('.', 1)[0]
            if name_prefix not in layer_list:
                param.requires_grad = False

        loss_list = []
        gradnorm_list = []

        for inputs, targets in data_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            predicts = F.softmax(outputs, dim=-1)
            log_predicts = torch.log(predicts)
            losses = F.nll_loss(log_predicts, targets, reduction='none')
            gradnorms = torch.empty(0).to(self.device)
            for loss in losses:
                self.model.zero_grad()
                loss.backward(retain_graph=True)
                grads = []
                for p in self.model.parameters():
                    if p.requires_grad:
                        per_sample_grad = p.grad.view(-1)
                        grads.append(per_sample_grad)
                grads = torch.cat(grads)
                # if torch.isnan(grads).any():
                #     print("Loss:", loss)
                #     print("NaN in gradients:", grads)
                #     print("Num of NaN:", torch.sum(torch.isnan(grads)))
                gradnorms = torch.cat((gradnorms, torch.linalg.norm(grads)[(None,)*2]), 0)
            losses = torch.unsqueeze(losses, 1)

            loss_list.append(losses.detach().data.cpu())
            gradnorm_list.append(gradnorms.detach().data.cpu())

        losses = torch.cat(loss_list, dim=0)
        gradnorms = torch.cat(gradnorm_list, dim=0)
        return losses, gradnorms
    
    def predict_target_confidence(self, data_loader):
        self.model.eval()

        confidence_list = []
        target_list = []
        for inputs, targets in data_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            predicts = F.softmax(outputs, dim=-1)
            confidence = torch.gather(predicts, 1, targets.unsqueeze(1))  
            max_val, _ = torch.max(confidence[confidence != 1], dim=0)  
            confidence[confidence >= 1] = max_val
            confidence_list.append(confidence.detach().cpu())
            target_list.append(targets.detach().cpu())

        confidences = torch.cat(confidence_list, dim=0)
        targets = torch.cat(target_list, dim=0)
        return confidences, targets
    
    def predict_target_lira_confidence(self, data_loader):
        self.model.eval()

        confidence_list = []
        target_list = []
        for inputs, targets in data_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            predicts = F.softmax(outputs, dim=-1)
            confidence = torch.gather(predicts, 1, targets.unsqueeze(1)) 
            max_val, _ = torch.max(confidence[confidence != 1], dim=0) 
            confidence[confidence >= 1] = max_val  
            min_val, _ = torch.min(confidence[confidence != 0], dim=0)
            confidence[confidence <= 0] = min_val 
            log_confidence = torch.log(confidence) - torch.log(1 - confidence)
            confidence_list.append(log_confidence.detach().cpu())
            target_list.append(targets.detach().cpu())

        confidences = torch.cat(confidence_list, dim=0)
        targets = torch.cat(target_list, dim=0)
        return confidences, targets
    
    def predict_target_loss(self, data_loader):
        self.model.eval()

        loss_list = []
        target_list = []
        for inputs, targets in data_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            predicts = F.softmax(outputs, dim=-1)
            log_predicts = torch.log(predicts)
            losses = F.nll_loss(log_predicts, targets, reduction='none')
            losses = torch.unsqueeze(losses, 1)
            loss_list.append(losses.detach().data.cpu())
            target_list.append(targets.detach().data.cpu())

        losses = torch.cat(loss_list, dim=0)
        targets = torch.cat(target_list, dim=0)
        return losses, targets
    
    def predict_target_ours_loss(self, data_loader):
        self.model.eval()

        loss_list = []
        target_list = []
        for inputs, targets in data_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            predicts = F.softmax(outputs, dim=-1)
            confidence = torch.gather(predicts, 1, targets.unsqueeze(1))  
            max_val, _ = torch.max(confidence[confidence != 1], dim=0) 
            confidence[confidence >= 1] = max_val  
            min_val, _ = torch.min(confidence[confidence != 0], dim=0)  
            confidence[confidence <= 0] = min_val 
            losses = torch.log(confidence) * (-1)
            losses = -1*losses + torch.pow(losses, -1/8)*8
            loss_list.append(losses.detach().data.cpu())
            target_list.append(targets.detach().data.cpu())

        losses = torch.cat(loss_list, dim=0)
        targets = torch.cat(target_list, dim=0)
        return losses, targets