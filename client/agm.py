import os
import pickle
import random
from argparse import Namespace
from collections import OrderedDict

import torch
from path import Path
from rich.console import Console
from rich.progress import track
from tqdm import tqdm

from config.utils import fix_random_seed, common_loss, calculate_prototype, EU_dist, process_grad, \
    evaluate_modality_acc, process_param_grad
from data.utils.util import get_train_dataset, get_val_dataset, CremadDataset
from models.basic_model import AVClassifier
from torch.utils.data import Subset, DataLoader
from copy import deepcopy
from typing import Dict, List, OrderedDict, Tuple
import numpy as np
import torch.nn as nn

import math

label_key = 'label'


class AGMClient:
    def __init__(
            self,
            args,
            model: torch.nn.Module,
            dataset: str,
            batch_size: int,
            local_epochs: int,
            local_lr: float,
            gpu: int, ):
        self.args = args
        self.device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

        self.client_id: int = None
        self.modality_type: str = None

        self.valset: Subset = None
        self.trainset: Subset = None
        self.testset: Subset = None

        self.model: torch.nn.Module = deepcopy(model).to(self.device)

        self.dataset = dataset
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.local_lr = local_lr
        self.criterion = torch.nn.CrossEntropyLoss()

        self.lr = 0

        self.global_epoch = 0
        self.audio_proto = {i: [] for i in range(self.args.client_num)}
        self.visual_proto = {i: [] for i in range(self.args.client_num)}

        self.visited_client = []

        if self.args.dataset == 'VGGSound':
            self.n_classes = 309
        elif self.args.dataset == 'KineticSound':
            self.n_classes = 31
        elif self.args.dataset == 'CREMAD':
            self.n_classes = 6
        elif self.args.dataset == 'AVE':
            self.n_classes = 28
        elif self.args.dataset == 'CGMNIST':
            self.n_classes = 10
        elif self.args.dataset == 'CrisisMMD':
            self.n_classes = 2
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(self.args.dataset))

        self.count_class = [[0 for _ in range(self.n_classes)] for _ in range(self.args.client_num)]

        self.iterations = [0 for _ in range(self.args.client_num)]

    @torch.no_grad()
    def evaluate(self, use_valset=True):
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = 0
        correct = 0
        dataloader = DataLoader(self.valset if use_valset else self.testset, 32)
        with torch.no_grad():
            for spec, image, y in dataloader:
                spec = spec.to(self.device)
                image = image.to(self.device)
                y = y.to(self.device)
                if self.dataset != 'CGMNIST':
                    spec, image = spec.unsqueeze(1).float(), image.float()

                _, _, logits = self.model(spec, image)
                loss += criterion(logits, y)
                pred = torch.softmax(logits, -1).argmax(-1)
                correct += (pred == y).int().sum()
        return loss.item() / len(dataloader), correct.item()

    def get_local_prototype(self, client_id, model):
        # self.client_id = client_id
        # self.get_client_local_dataset()
        # train_dataloader = DataLoader(self.trainset, batch_size=self.batch_size,
        #                               shuffle=True, pin_memory=False)  # 计算机的内存充足的时候，可以设置pin_memory=True
        # audio_proto, visual_proto = calculate_prototype(self.args, model, train_dataloader, self.device, epoch=0)

        return self.audio_proto[client_id], self.visual_proto[client_id], self.count_class[client_id]

    def get_local_loss(self, client_id, model, modality_type=None):
        self.client_id = client_id
        self.modality_type = modality_type
        # self.model.load_state_dict(model.state_dict())
        self.get_client_local_dataset()

        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)
        relu = nn.ReLU(inplace=True)
        tanh = nn.Tanh()

        train_dataloader = DataLoader(self.trainset, batch_size=self.batch_size,
                                      shuffle=True, pin_memory=False)  # 计算机的内存充足的时候，可以设置pin_memory=True
        local_loss = 0
        n = 0
        with torch.no_grad():
            model.eval()
            for step, (spec, image, label) in enumerate(train_dataloader):
                spec = spec.to(self.device)  # B x 257 x 1004(CREMAD 299)
                image = image.to(self.device)  # B x 1(image count) x 3 x 224 x 224
                label = label.to(self.device)  # B
                if self.modality_type == 'multimodal':
                    # TODO: make it simpler and easier to extend
                    if self.args.dataset != 'CGMNIST':
                        a, v, out = self.model(spec.unsqueeze(1).float(), image.float())
                    else:
                        a, v, out = self.model(spec, image)  # gray colored
                elif self.modality_type == 'audio':
                    if self.args.dataset != 'CGMNIST':
                        a, out = self.model.forward_audio(spec.unsqueeze(1).float())
                    else:
                        a, out = self.model.forward_gray(spec)
                elif self.modality_type == 'visual':
                    if self.args.dataset != 'CGMNIST':
                        v, out = self.model.forward_visual(image.float(), bsz=label.shape[0])
                    else:
                        v, out = self.model.forward_colored(image)
                else:
                    raise ValueError('No such modality type.')
                loss = criterion(out, label)
                local_loss += loss.item()
                n += 1
        return local_loss / n

    def train(
            self,
            client_id: int,
            model,
            modality_type=None,
            global_epoch=0,
            last_score_a=None,
            last_score_v=None,
            audio_lr_ratio=None,
            visual_lr_ratio=None

    ):
        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)
        relu = nn.ReLU(inplace=True)
        tanh = nn.Tanh()
        self.client_id = client_id
        self.modality_type = modality_type
        self.model.load_state_dict(model.state_dict())
        self.model.train()
        self.get_client_local_dataset()
        train_dataloader = DataLoader(self.trainset, batch_size=self.batch_size,
                                      shuffle=True, pin_memory=False)  # 计算机的内存充足的时候，可以设置pin_memory=True
        total_batch = len(train_dataloader)

        if self.args.dataset == 'CREMAD':
            n_classes = 6
        else:
            raise NotImplementedError('Incorrect dataset name')

        self.global_epoch = global_epoch

        self.model.mode = 'train'
        if modality_type == "audio":
            train_score_a = last_score_a
            ra_score_a = 0.
        elif modality_type == "visual":
            train_score_v = last_score_v
            ra_score_v = 0.
        else:
            train_score_a = last_score_a
            train_score_v = last_score_v
            ra_score_a = 0.
            ra_score_v = 0.
        train_batch_loss = 0.

        if self.global_epoch > 60:
            self.lr = self.args.local_lr - (
                        self.args.local_lr - self.args.lr_min) / self.args.global_epochs * global_epoch
        else:
            self.lr = self.args.local_lr

        optimizer: torch.optim.Optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.args.lr_decay_step, self.args.lr_decay_ratio)

        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        for le in range(self.local_epochs):
            for step, (spec, image, label) in enumerate(train_dataloader):
                spec = spec.to(self.device)
                image = image.to(self.device)
                label = label.to(self.device)
                self.iterations[self.client_id] += 1
                # iteration = (self.global_epoch*self.local_epochs+le) * total_batch + step + 1

                if modality_type == "audio":
                    if self.args.fusion_type == "early_fusion":
                        out_a = self.model.net(spec.unsqueeze(1).float(), image.float(), pad_audio=False, pad_visual=True)
                    elif self.args.fusion_type == "late_fusion":
                        out_a = self.model(spec.unsqueeze(1))
                    loss = criterion(out_a, label)
                elif modality_type == "visual":
                    if self.args.fusion_type == "early_fusion":
                        out_v = self.model.net(spec.unsqueeze(1).float(), image.float(), pad_audio=True, pad_visual=False)
                    elif self.argss.fusion_type == "late_fusion":
                        out_v = self.model(image.float())
                    loss = criterion(out_v, label)
                else:
                    if self.args.fusion_type == "early_fusion" or self.args.fl_method == "AGM":
                        total_out, pad_visual_out, pad_audio_out, zero_padding_out, out = self.model(spec.unsqueeze(1).float(),
                                                                                                image.float())
                        out_a = 0.5 * (total_out - pad_audio_out + pad_visual_out)
                        out_v = 0.5 * (total_out - pad_visual_out + pad_audio_out)
                    else:
                        out_a, out_v, out = self.model(spec.unsqueeze(1).float(), image.float())
                    loss = criterion(out, label)

                train_batch_loss += loss.item() / total_batch

                if modality_type == 'audio':
                    pred_a = softmax(out_a)
                elif modality_type == 'visual':
                    pred_v = softmax(out_v)
                else:
                    prediction = softmax(out)
                    pred_a = softmax(out_a)
                    pred_v = softmax(out_v)

                for j in range(image.shape[0]):
                    if modality_type == 'audio':
                        a = np.argmax(pred_a[j].cpu().data.numpy())
                        num[label[j]] += 1.0
                        if np.asarray(label[j].cpu()) == a:
                            acc_a[label[j]] += 1.0
                    elif modality_type == 'visual':
                        v = np.argmax(pred_v[j].cpu().data.numpy())
                        num[label[j]] += 1.0
                        if np.asarray(label[j].cpu()) == v:
                            acc_v[label[j]] += 1.0
                    else:
                        ma = np.argmax(prediction[j].cpu().data.numpy())
                        v = np.argmax(pred_v[j].cpu().data.numpy())
                        a = np.argmax(pred_a[j].cpu().data.numpy())
                        num[label[j]] += 1.0

                        if np.asarray(label[j].cpu()) == ma:
                            acc[label[j]] += 1.0
                        if np.asarray(label[j].cpu()) == v:
                            acc_v[label[j]] += 1.0
                        if np.asarray(label[j].cpu()) == a:
                            acc_a[label[j]] += 1.0

                optimizer.zero_grad()

                if modality_type == "audio":
                    score_audio = 0.
                    for k in range(out_a.size(0)):
                        score_audio += - torch.log(softmax(out_a)[k][label[k]])

                    score_audio = score_audio / out_a.size(0)
                    train_score_a = train_score_a * (self.iterations[self.client_id] - 1) / self.iterations[self.client_id] + score_audio.item() / self.iterations[self.client_id]
                    loss.backward()
                elif modality_type == "visual":
                    score_visual = 0.
                    for k in range(out_v.size(0)):
                        score_visual += - torch.log(softmax(out_v)[k][label[k]])
                    score_visual = score_visual / out_v.size(0)
                    train_score_v = train_score_v * (self.iterations[self.client_id] - 1) / self.iterations[self.client_id] + score_visual.item() / self.iterations[self.client_id]
                    loss.backward()
                else:
                    if torch.isnan(out_a).any() or torch.isnan(out_v).any():
                        raise ValueError

                    score_audio = 0.
                    score_visual = 0.
                    for k in range(out_a.size(0)):
                        if torch.isinf(torch.log(softmax(out_a)[k][label[k]])) or softmax(out_a)[k][label[k]] < 1e-8:
                            score_audio += - torch.log(torch.tensor(1e-8, dtype=out_a.dtype, device=out_a.device))
                        else:
                            score_audio += - torch.log(softmax(out_a)[k][label[k]])

                        if torch.isinf(torch.log(softmax(out_v)[k][label[k]])) or softmax(out_v)[k][label[k]] < 1e-8:
                            score_visual += - torch.log(torch.tensor(1e-8, dtype=out_v.dtype, device=out_v.device))
                        else:
                            score_visual += - torch.log(softmax(out_v)[k][label[k]])
                    score_audio = score_audio / out_a.size(0)
                    score_visual = score_visual / out_v.size(0)

                    # ratio_a = math.exp(score_visual.item() - score_audio.item())  # different from the formula in paper
                    # ratio_v = math.exp(score_audio.item() - score_visual.item())

                    ratio_v = math.exp(score_visual.item() - score_audio.item())  # corrected version
                    ratio_a = math.exp(score_audio.item() - score_visual.item())

                    # optimal_ratio_a = math.exp(train_score_v - train_score_a)
                    # optimal_ratio_v = math.exp(train_score_a - train_score_v)

                    optimal_ratio_a = math.exp(train_score_a - train_score_v)
                    optimal_ratio_v = math.exp(train_score_v - train_score_a)

                    coeff_a = math.exp(self.args.MI_alpha * (min(optimal_ratio_a - ratio_a, 10)))
                    coeff_v = math.exp(self.args.MI_alpha * (min(optimal_ratio_v - ratio_v, 10)))

                    train_score_a = train_score_a * (self.iterations[self.client_id] - 1) / self.iterations[self.client_id] + score_audio.item() / self.iterations[self.client_id]  # the iteration should be client-wise
                    train_score_v = train_score_v * (self.iterations[self.client_id] - 1) / self.iterations[self.client_id] + score_visual.item() / self.iterations[self.client_id]
                    ra_score_a = ra_score_a * step / (step + 1) + score_audio.item() / (step + 1)
                    ra_score_v = ra_score_v * step / (step + 1) + score_visual.item() / (step + 1)

                    if self.args.fl_method == "AGM" and self.args.modulation_starts <= self.global_epoch <= self.args.modulation_ends:
                        self.model.update_scale(coeff_a, coeff_v)
                        loss.backward()

                    elif self.args.fl_method == "MSLR" and self.args.modulation_starts <= self.global_epoch <= self.args.modulation_ends:
                        audio_lr_init_coeff = 0.9
                        visual_lr_init_coeff = 1.1
                        mslr_audio_coeff = audio_lr_init_coeff * audio_lr_ratio
                        mslr_visual_coeff = visual_lr_init_coeff * visual_lr_ratio
                        loss.backward()
                        for name, params in self.model.named_parameters():
                            layer = str(name).split('.')[0]
                            if 'audio_net' in layer or 'audio_cls' in layer:
                                params.grad = params.grad * mslr_audio_coeff

                            if 'visual_net' in layer or 'visual_cls' in layer:
                                params.grad = params.grad * mslr_visual_coeff

                    elif self.args.fl_method == "MSES" and self.args.modulation_starts <= self.global_epoch <= self.args.modulation_ends:
                        audio_lr_init_coeff = 1.0
                        visual_lr_init_coeff = 1.0
                        mses_audio_coeff = audio_lr_init_coeff * audio_lr_ratio
                        mses_visual_coeff = visual_lr_init_coeff * visual_lr_ratio
                        loss.backward()
                        for name, params in self.model.named_parameters():
                            layer = str(name).split('.')[0]
                            if 'audio_net' in layer or 'audio_cls' in layer:
                                params.grad = params.grad * mses_audio_coeff

                            if 'visual_net' in layer or 'visual_cls' in layer:
                                params.grad = params.grad * mses_visual_coeff
                    else:
                        #
                        loss.backward()
                    if self.args.fl_method == "AGM" or self.args.fusion_type == "early_fusion":
                        if self.args.fusion_type == "late_fusion":
                            grad_max = torch.max(self.model.net.fusion_module.fc_out.weight.grad)
                            grad_min = torch.min(self.model.net.fusion_module.fc_out.weight.grad)
                            if grad_max > 1 or grad_min < -1:
                                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        else:
                            grad_max = torch.max(self.model.net.head.fc.weight.grad)
                            grad_min = torch.min(self.model.net.head.fc.weight.grad)
                            if grad_max > 1 or grad_min < -1:
                                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

        all_params = self.model.state_dict()
        data_num_client = self.trainset.data_num

        if modality_type == "audio":
            accuracy_a = sum(acc_a) / sum(num)
            return train_score_a, last_score_v, all_params, data_num_client
        elif modality_type == "visual":
            accuracy_v = sum(acc_v) / sum(num)
            return last_score_a, train_score_v, all_params, data_num_client
        else:
            accuracy = sum(acc) / sum(num)
            accuracy_a = sum(acc_a) / sum(num)
            accuracy_v = sum(acc_v) / sum(num)
            return train_score_a, train_score_v, all_params, data_num_client


    def _train(self):
        optimizer: torch.optim.Optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.args.lr_decay_step, self.args.lr_decay_ratio)

        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)
        relu = nn.ReLU(inplace=True)
        tanh = nn.Tanh()

        train_dataloader = DataLoader(self.trainset, batch_size=self.batch_size,
                                      shuffle=True, pin_memory=False)  # 计算机的内存充足的时候，可以设置pin_memory=True

        if self.args.PMR and self.args.modulation_starts <= self.global_epoch <= self.args.modulation_ends:
            self.audio_proto[self.client_id], self.visual_proto[self.client_id] = calculate_prototype(self.args,
                                                                                                      self.model,
                                                                                                      train_dataloader,
                                                                                                      self.device,
                                                                                                      epoch=0, ratio=1.0)

        for le in range(self.local_epochs):
            self.model.train()
            _loss = 0

            if self.args.dataset == 'CrisisMMD':
                for step, data in enumerate(train_dataloader):
                    x = (data['image'].to(self.device),
                         {k: v.to(self.device) for k, v in data['text_tokens'].items()})
                    label = data[label_key].to(self.device)

                    optimizer.zero_grad()
                    if self.modality_type == 'multimodal':
                        # TODO: make it simpler and easier to extend
                        a, v, out = self.model(x)  # image, text
                    elif self.modality_type == 'audio':
                        a, out = self.model.forward_image(x)
                    elif self.modality_type == 'visual':
                        v, out = self.model.forward_text(x)
                    else:
                        raise ValueError('No such modality type.')

                    if self.args.PMR and self.args.modulation_starts <= self.global_epoch <= self.args.modulation_ends and self.modality_type == 'multimodal':
                        if self.args.PMR:
                            # loss with local proto
                            audio_sim = -EU_dist(a, self.audio_proto[self.client_id])  # B x n_class
                            visual_sim = -EU_dist(v, self.visual_proto[self.client_id])  # B x n_class

                            score_a_p = sum([softmax(audio_sim)[i][label[i]] for i in range(audio_sim.size(0))])
                            score_v_p = sum([softmax(visual_sim)[i][label[i]] for i in range(visual_sim.size(0))])
                            ratio_a_p = score_a_p / score_v_p
                            loss_proto_a = criterion(audio_sim, label)
                            loss_proto_v = criterion(visual_sim, label)

                            if ratio_a_p > 1:
                                beta = 0  # audio coef
                                lam = 1 * self.args.MI_alpha  # visual coef
                            elif ratio_a_p < 1:
                                beta = 1 * self.args.MI_alpha
                                lam = 0
                            else:
                                beta = 0
                                lam = 0
                            # print(loss_proto_a, loss_proto_v)
                            loss = criterion(out, label) + beta * loss_proto_a + lam * loss_proto_v
                    elif self.args.fedproto and self.args.modulation_starts <= self.global_epoch <= self.args.modulation_ends and self.global_epoch >= 1:
                        # loss with global proto
                        if self.modality_type == 'multimodal':
                            audio_sim_global = -EU_dist(a, self.global_audio_proto)  # B x n_class
                            visual_sim_global = -EU_dist(v, self.global_visual_proto)  # B x n_class
                            loss_proto_a = criterion(audio_sim_global, label)
                            loss_proto_v = criterion(visual_sim_global, label)
                        elif self.modality_type == 'audio':
                            audio_sim_global = -EU_dist(a, self.global_audio_proto)  # B x n_class
                            loss_proto_a = criterion(audio_sim_global, label)
                        elif self.modality_type == 'visual':
                            visual_sim_global = -EU_dist(v, self.global_visual_proto)  # B x n_class
                            loss_proto_v = criterion(visual_sim_global, label)

                        if self.modality_type == 'multimodal':
                            loss = criterion(out, label) + self.args.MI_alpha * (loss_proto_a + loss_proto_v)
                        elif self.modality_type == 'audio':
                            loss = criterion(out, label) + self.args.MI_alpha * loss_proto_a
                        elif self.modality_type == 'visual':
                            loss = criterion(out, label) + self.args.MI_alpha * loss_proto_v
                        # loss = criterion(out, label) + beta * loss_proto_a + lam * loss_proto_v + loss_common
                    else:
                        loss = criterion(out, label)
                    # print('loss: ', loss, loss_common)
                    _loss += loss.item()
                    loss.backward()

                    if self.args.OGM and self.args.modulation_starts <= self.global_epoch <= self.args.modulation_ends and self.modality_type == 'multimodal':
                        weight_size = self.model.cls_layer.weight.size(1)
                        out_v = (torch.mm(v, torch.transpose(self.model.cls_layer.weight[:, self.args.dim_img_repr:], 0, 1))
                                 + self.model.cls_layer.bias / 2)
                        out_a = (torch.mm(a, torch.transpose(self.model.cls_layer.weight[:, :self.args.dim_img_repr], 0, 1))
                                 + self.model.cls_layer.bias / 2)

                        score_v = sum([softmax(out_v)[i][label[i]] for i in range(out_v.size(0))])
                        score_a = sum([softmax(out_a)[i][label[i]] for i in range(out_a.size(0))])
                        ratio_v = score_v / score_a
                        ratio_a = 1 / ratio_v

                        """
                        Below is the Eq.(10) in our CVPR paper:
                                1 - tanh(alpha * rho_t_u), if rho_t_u > 1
                        k_t_u =
                                1,                         else
                        coeff_u is k_t_u, where t means iteration steps and u is modality indicator, either a or v.
                        """
                        if ratio_v > 1:
                            coeff_v = 1 - tanh(self.args.MI_alpha * relu(ratio_v))
                            coeff_a = 1
                            # acc_v = 1
                            # acc_a = 1 + tanh(self.args.MI_alpha * relu(ratio_v))
                        else:
                            coeff_a = 1 - tanh(self.args.MI_alpha * relu(ratio_a))
                            coeff_v = 1
                            # acc_a = 1
                            # acc_v = 1 + tanh(self.args.MI_alpha * relu(ratio_a))

                        for name, parms in self.model.named_parameters():
                            layer = str(name).split('.')[0]
                            if self.args.dataset != 'CGMNIST':
                                if 'audio' in layer and len(parms.grad.size()) == 4:
                                    parms.grad = parms.grad * coeff_a + \
                                                 torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)

                                if 'visual' in layer and len(parms.grad.size()) == 4:
                                    parms.grad = parms.grad * coeff_v + \
                                                 torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                            else:
                                if 'gray' in layer and len(parms.grad.size()) == 4:
                                    parms.grad = parms.grad * coeff_a + \
                                                 torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)

                                if 'colored' in layer and len(parms.grad.size()) == 4:
                                    parms.grad = parms.grad * coeff_v + \
                                                 torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)

                    optimizer.step()
            else:
                for step, (spec, image, label) in enumerate(train_dataloader):
                    spec = spec.to(self.device)  # B x 257 x 1004(CREMAD 299)
                    image = image.to(self.device)  # B x 1(image count) x 3 x 224 x 224
                    label = label.to(self.device)  # B

                    optimizer.zero_grad()

                    if self.modality_type == 'multimodal':
                        # TODO: make it simpler and easier to extend
                        if self.args.dataset != 'CGMNIST':
                            a, v, out = self.model(spec.unsqueeze(1).float(), image.float())
                        else:
                            a, v, out = self.model(spec, image)  # gray colored
                    elif self.modality_type == 'audio':
                        if self.args.dataset != 'CGMNIST':
                            a, out = self.model.forward_audio(spec.unsqueeze(1).float())
                        else:
                            a, out = self.model.forward_gray(spec)
                    elif self.modality_type == 'visual':
                        if self.args.dataset != 'CGMNIST':
                            v, out = self.model.forward_visual(image.float(), bsz=label.shape[0])
                        else:
                            v, out = self.model.forward_colored(image)
                    else:
                        raise ValueError('No such modality type.')

                    if self.args.PMR and self.args.modulation_starts <= self.global_epoch <= self.args.modulation_ends and self.modality_type == 'multimodal':
                        if self.args.PMR:
                            # loss with local proto
                            audio_sim = -EU_dist(a, self.audio_proto[self.client_id])  # B x n_class
                            visual_sim = -EU_dist(v, self.visual_proto[self.client_id])  # B x n_class

                            score_a_p = sum([softmax(audio_sim)[i][label[i]] for i in range(audio_sim.size(0))])
                            score_v_p = sum([softmax(visual_sim)[i][label[i]] for i in range(visual_sim.size(0))])
                            ratio_a_p = score_a_p / score_v_p
                            loss_proto_a = criterion(audio_sim, label)
                            loss_proto_v = criterion(visual_sim, label)

                            # loss with global proto
                            # ratio_a_p = 1.5
                            # audio_sim_global = -EU_dist(a, self.global_audio_proto)  # B x n_class
                            # visual_sim_global = -EU_dist(v, self.global_visual_proto)  # B x n_class
                            # loss_proto_a = criterion(audio_sim_global, label)
                            # loss_proto_v = criterion(visual_sim_global, label)

                            if ratio_a_p > 1:
                                beta = 0  # audio coef
                                lam = 1 * self.args.MI_alpha  # visual coef
                            elif ratio_a_p < 1:
                                beta = 1 * self.args.MI_alpha
                                lam = 0
                            else:
                                beta = 0
                                lam = 0
                            # print(loss_proto_a, loss_proto_v)
                            loss = criterion(out, label) + beta * loss_proto_a + lam * loss_proto_v
                    elif self.args.fedproto and self.args.modulation_starts <= self.global_epoch <= self.args.modulation_ends and self.global_epoch >= 1:
                        # loss with global proto
                        if self.modality_type == 'multimodal':
                            audio_sim_global = -EU_dist(a, self.global_audio_proto)  # B x n_class
                            visual_sim_global = -EU_dist(v, self.global_visual_proto)  # B x n_class
                            loss_proto_a = criterion(audio_sim_global, label)
                            loss_proto_v = criterion(visual_sim_global, label)
                        elif self.modality_type == 'audio':
                            audio_sim_global = -EU_dist(a, self.global_audio_proto)  # B x n_class
                            loss_proto_a = criterion(audio_sim_global, label)
                        elif self.modality_type == 'visual':
                            visual_sim_global = -EU_dist(v, self.global_visual_proto)  # B x n_class
                            loss_proto_v = criterion(visual_sim_global, label)

                        if self.modality_type == 'multimodal':
                            loss = criterion(out, label) + self.args.MI_alpha * (loss_proto_a + loss_proto_v)
                        elif self.modality_type == 'audio':
                            loss = criterion(out, label) + self.args.MI_alpha * loss_proto_a
                        elif self.modality_type == 'visual':
                            loss = criterion(out, label) + self.args.MI_alpha * loss_proto_v
                        # loss = criterion(out, label) + beta * loss_proto_a + lam * loss_proto_v + loss_common
                    else:
                        loss = criterion(out, label)
                    # print('loss: ', loss, loss_common)
                    _loss += loss.item()
                    loss.backward()

                    if self.args.OGM and self.args.modulation_starts <= self.global_epoch <= self.args.modulation_ends and self.modality_type == 'multimodal':
                        if self.args.fusion_method == 'sum':
                            out_v = (torch.mm(v, torch.transpose(self.model.fusion_module.fc_y.weight, 0, 1)) +
                                     self.model.fusion_module.fc_y.bias)
                            out_a = (torch.mm(a, torch.transpose(self.model.fusion_module.fc_x.weight, 0, 1)) +
                                     self.model.fusion_module.fc_x.bias)
                        elif self.args.fusion_method == 'concat':
                            weight_size = self.model.fusion_module.fc_out.weight.size(1)
                            out_v = (torch.mm(v, torch.transpose(self.model.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
                                     + self.model.fusion_module.fc_out.bias / 2)
                            out_a = (torch.mm(a, torch.transpose(self.model.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
                                     + self.model.fusion_module.fc_out.bias / 2)
                        elif self.args.fusion_method == 'film':
                            out_v = out
                            out_a = out

                        score_v = sum([softmax(out_v)[i][label[i]] for i in range(out_v.size(0))])
                        score_a = sum([softmax(out_a)[i][label[i]] for i in range(out_a.size(0))])
                        ratio_v = score_v / score_a
                        ratio_a = 1 / ratio_v

                        """
                        Below is the Eq.(10) in our CVPR paper:
                                1 - tanh(alpha * rho_t_u), if rho_t_u > 1
                        k_t_u =
                                1,                         else
                        coeff_u is k_t_u, where t means iteration steps and u is modality indicator, either a or v.
                        """
                        if ratio_v > 1:
                            coeff_v = 1 - tanh(self.args.MI_alpha * relu(ratio_v))
                            coeff_a = 1
                            # acc_v = 1
                            # acc_a = 1 + tanh(self.args.MI_alpha * relu(ratio_v))
                        else:
                            coeff_a = 1 - tanh(self.args.MI_alpha * relu(ratio_a))
                            coeff_v = 1
                            # acc_a = 1
                            # acc_v = 1 + tanh(self.args.MI_alpha * relu(ratio_a))

                        for name, parms in self.model.named_parameters():
                            layer = str(name).split('.')[0]
                            # print('params: ', parms)
                            # print('name: ', name)
                            # print('layer: ', layer)
                            if self.args.dataset != 'CGMNIST':
                                if 'audio' in layer and len(parms.grad.size()) == 4:
                                    parms.grad = parms.grad * coeff_a + \
                                                 torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)

                                if 'visual' in layer and len(parms.grad.size()) == 4:
                                    parms.grad = parms.grad * coeff_v + \
                                                 torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                            else:
                                if 'gray' in layer and len(parms.grad.size()) == 4:
                                    parms.grad = parms.grad * coeff_a + \
                                                 torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)

                                if 'colored' in layer and len(parms.grad.size()) == 4:
                                    parms.grad = parms.grad * coeff_v + \
                                                 torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)

                    optimizer.step()

            if self.args.optimizer == 'SGD':
                scheduler.step()

            if self.args.PMR and self.args.modulation_starts <= self.global_epoch <= self.args.modulation_ends:
                self.audio_proto[self.client_id], self.visual_proto[self.client_id] = calculate_prototype(self.args,
                                                                                                          self.model,
                                                                                                          train_dataloader,
                                                                                                          self.device,
                                                                                                          epoch=0, ratio=1.0)
        if self.args.fedproto and self.args.modulation_starts <= self.global_epoch <= self.args.modulation_ends:
            self.audio_proto[self.client_id], self.visual_proto[self.client_id] = calculate_prototype(self.args,
                                                                                                      self.model,
                                                                                                      train_dataloader,
                                                                                                      self.device,
                                                                                                      epoch=0, ratio=1.0)


        print('client {} loss: '.format(self.client_id), _loss / len(train_dataloader))

        if self.args.dataset == 'CREMAD' or self.args.dataset == 'AVE':
            model_params = [self.model.audio_net.state_dict(), self.model.visual_net.state_dict(),
                            self.model.fusion_module.state_dict()]
        elif self.args.dataset == 'CGMNIST':
            model_params = [self.model.gray_net.state_dict(), self.model.colored_net.state_dict(),
                            self.model.fusion_module.state_dict()]
        elif self.args.dataset == 'CrisisMMD':
            model_params = [self.model.imageEncoder.state_dict(), self.model.textEncoder.state_dict(),
                            self.model.cls_layer.state_dict()]
        else:
            raise ValueError('wrong dataset name.')

        return model_params, self.trainset.data_num

    def validate(self):
        val_dataset = get_val_dataset(self.args, self.args.dataset)
        val_dataloader = DataLoader(val_dataset, batch_size=self.args.val_batch_size, shuffle=False, pin_memory=False)

        if self.args.dataset == 'VGGSound':
            n_classes = 309
        elif self.args.dataset == 'KineticSound':
            n_classes = 31
        elif self.args.dataset == 'CREMAD':
            n_classes = 6
        elif self.args.dataset == 'AVE':
            n_classes = 28
        elif self.args.dataset == 'CGMNIST':
            n_classes = 10
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(self.args.dataset))

        self.model.eval()
        softmax = nn.Softmax(dim=1)
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]

        with torch.no_grad():
            for step, (spec, image, label) in enumerate(val_dataloader):
                spec = spec.to(self.device)
                image = image.to(self.device)
                label = label.to(self.device)

                if self.args.dataset != 'CGMNIST':
                    a, v, out = self.model(spec.unsqueeze(1).float(), image.float())
                else:
                    a, v, out = self.model(spec, image)  # gray colored

                prediction = softmax(out)
                for i in range(image.shape[0]):

                    ma = np.argmax(prediction[i].cpu().data.numpy())
                    num[label[i]] += 1.0
                    # pdb.set_trace()
                    if np.asarray(label[i].cpu()) == ma:
                        acc[label[i]] += 1.0
        return sum(acc) / sum(num)

    def get_data_batch(self):
        batch_size = (
            self.batch_size
            if self.batch_size > 0
            else int(len(self.trainset) / self.local_epochs)
        )
        indices = torch.from_numpy(
            np.random.choice(self.trainset.indices, batch_size)
        ).long()
        spec, image, targets = self.trainset.dataset[indices]
        return spec.to(self.device), image.to(self.device), targets.to(self.device)

    def get_client_local_dataset(self):
        if self.args.dataset == 'VGGSound':
            n_classes = 309
        elif self.args.dataset == 'KineticSound':
            n_classes = 31
        elif self.args.dataset == 'CREMAD':
            n_classes = 6
        elif self.args.dataset == 'AVE':
            n_classes = 28
        elif self.args.dataset == 'CGMNIST':
            n_classes = 10
        elif self.args.dataset == 'CrisisMMD':
            n_classes = 2
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(self.args.dataset))

        self.count_class[self.client_id] = [0 for _ in range(n_classes)]

        datasets = get_train_dataset(
            self.args,
            self.dataset,
            self.client_id,
        )
        self.trainset = datasets

        dataloader = DataLoader(self.trainset, batch_size=self.batch_size,
                                shuffle=False, pin_memory=False)  # 计算机的内存充足的时候，可以设置pin_memory=True

        if self.args.dataset != 'CrisisMMD':
            for step, (_, _, label) in enumerate(dataloader):
                label = label.to(self.device)  # B
                for c, l in enumerate(label):
                    l = l.long()
                    self.count_class[self.client_id][l] += 1
        else:
            for step, data in enumerate(dataloader):
                label = data['label'].to(self.device)
                for c, l in enumerate(label):
                    l = l.long()
                    self.count_class[self.client_id][l] += 1

    def set_parameters(self, model_params):
        audio_params, visual_params, fusion_params = model_params[0], model_params[1], model_params[2]
        # print('audio: ', audio_params)
        self.model.audio_net.load_state_dict(audio_params, strict=False)
        self.model.visual_net.load_state_dict(visual_params, strict=False)
        if self.args.fusion_method == 'concat':
            self.model.fusion_module.load_state_dict(fusion_params, strict=False)
        # self.model.load_state_dict(model_params, strict=False)
        # if self.client_id in self.untrainable_params.keys():
        #     self.model.load_state_dict(
        #         self.untrainable_params[self.client_id], strict=False
        #     )

    def get_grads(self, client_id, model, modality_type, model_len):
        self.client_id = client_id
        self.modality_type = modality_type
        self.model.load_state_dict(model.state_dict())
        self.get_client_local_dataset()

        optimizer: torch.optim.Optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)

        criterion = nn.CrossEntropyLoss()

        train_dataloader = DataLoader(self.trainset, batch_size=self.batch_size,
                                      shuffle=True, pin_memory=False)  # 计算机的内存充足的时候，可以设置pin_memory=True

        for le in range(self.args.grad_epochs):
            self.model.train()
            for step, (spec, image, label) in enumerate(train_dataloader):
                spec = spec.to(self.device)  # B x 257 x 1004(CREMAD 299)
                image = image.to(self.device)  # B x 1(image count) x 3 x 224 x 224
                label = label.to(self.device)  # B

                optimizer.zero_grad()

                if self.modality_type == 'multimodal':
                    # TODO: make it simpler and easier to extend
                    if self.args.dataset != 'CGMNIST':
                        a, v, out = self.model(spec.unsqueeze(1).float(), image.float())
                    else:
                        a, v, out = self.model(spec, image)  # gray colored
                elif self.modality_type == 'audio':
                    if self.args.dataset != 'CGMNIST':
                        a, out = self.model.forward_audio(spec.unsqueeze(1).float())
                    else:
                        a, out = self.model.forward_gray(spec)
                elif self.modality_type == 'visual':
                    if self.args.dataset != 'CGMNIST':
                        v, out = self.model.forward_visual(image.float(), bsz=label.shape[0])
                    else:
                        v, out = self.model.forward_colored(image)
                else:
                    raise ValueError('No such modality type.')

                loss = criterion(out, label)
                loss.backward()
                optimizer.step()

        global_params = process_grad(model.parameters())
        local_params = process_grad(self.model.parameters())
        client_grads = global_params - local_params
        return self.trainset.data_num, client_grads

    def get_modal_grads(self, client_id, model, modality_type, model_len):
        self.client_id = client_id
        self.modality_type = modality_type
        self.model.load_state_dict(model.state_dict())
        self.get_client_local_dataset()

        optimizer: torch.optim.Optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)

        criterion = nn.CrossEntropyLoss()

        train_dataloader = DataLoader(self.trainset, batch_size=self.batch_size,
                                      shuffle=True, pin_memory=False)  # 计算机的内存充足的时候，可以设置pin_memory=True

        for le in range(self.args.grad_epochs):
            self.model.train()
            for step, (spec, image, label) in enumerate(train_dataloader):
                spec = spec.to(self.device)  # B x 257 x 1004(CREMAD 299)
                image = image.to(self.device)  # B x 1(image count) x 3 x 224 x 224
                label = label.to(self.device)  # B

                optimizer.zero_grad()

                if self.modality_type == 'multimodal':
                    # TODO: make it simpler and easier to extend
                    if self.args.dataset != 'CGMNIST':
                        a, v, out = self.model(spec.unsqueeze(1).float(), image.float())
                    else:
                        a, v, out = self.model(spec, image)  # gray colored
                elif self.modality_type == 'audio':
                    if self.args.dataset != 'CGMNIST':
                        a, out = self.model.forward_audio(spec.unsqueeze(1).float())
                    else:
                        a, out = self.model.forward_gray(spec)
                elif self.modality_type == 'visual':
                    if self.args.dataset != 'CGMNIST':
                        v, out = self.model.forward_visual(image.float(), bsz=label.shape[0])
                    else:
                        v, out = self.model.forward_colored(image)
                else:
                    raise ValueError('No such modality type.')

                loss = criterion(out, label)
                loss.backward()
                optimizer.step()

        acc_a, acc_v = evaluate_modality_acc(self.args, self.model, train_dataloader, self.device, epoch=0, ratio=1.0)
        b_ratio = acc_a / acc_v  # modality imbalance ratio

        global_params = process_grad(model.parameters())
        local_params = process_grad(self.model.parameters())
        client_grads = global_params - local_params

        if self.args.dataset != 'CGMNIST':
            global_params_a = process_grad(model.audio_net.parameters())
            local_params_a = process_grad(self.model.audio_net.parameters())
            global_params_v = process_grad(model.visual_net.parameters())
            local_params_v = process_grad(self.model.visual_net.parameters())
        else:
            global_params_a = process_grad(model.gray_net.parameters())
            local_params_a = process_grad(self.model.gray_net.parameters())
            global_params_v = process_grad(model.colored_net.parameters())
            local_params_v = process_grad(self.model.colored_net.parameters())
        client_grads_a = global_params_a - local_params_a

        client_grads_v = global_params_v - local_params_v

        self.model.load_state_dict(model.state_dict())  # 重置
        return self.trainset.data_num, client_grads, client_grads_a, client_grads_v, b_ratio

    def get_PMR_grads(self, client_id, model, modality_type, model_len):
        """
        generate gradient before the first global round
        :param client_id:
        :param model:
        :param modality_type:
        :param model_len: mm, a, v
        :return:
        """
        softmax = nn.Softmax(dim=1)

        self.client_id = client_id
        self.modality_type = modality_type
        param_len, param_len_a, param_len_v = model_len[0], model_len[1], model_len[2]
        self.model.load_state_dict(model.state_dict())
        self.get_client_local_dataset()

        # global_grads = np.zeros(param_len_a + param_len_v)
        # audio_grads = np.zeros(param_len_a)
        # visual_grads = np.zeros(param_len_v)

        optimizer: torch.optim.Optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)

        criterion = nn.CrossEntropyLoss()

        train_dataloader = DataLoader(self.trainset, batch_size=self.batch_size,
                                      shuffle=True, pin_memory=False)  # 计算机的内存充足的时候，可以设置pin_memory=True

        audio_proto, visual_proto = calculate_prototype(self.args, self.model, train_dataloader, self.device, epoch=-1, ratio=1.0,
                                                        a_proto=None, v_proto=None)
        self.audio_proto[client_id] = audio_proto
        self.visual_proto[client_id] = visual_proto

        for le in range(self.args.grad_epochs):
            self.model.train()
            if self.args.dataset == 'CrisisMMD':
                for step, data in enumerate(train_dataloader):
                    x = (data['image'].to(self.device),
                         {k: v.to(self.device) for k, v in data['text_tokens'].items()})
                    label = data[label_key].to(self.device)

                    optimizer.zero_grad()

                    if self.modality_type == 'multimodal':
                        # TODO: make it simpler and easier to extend
                        a, v, out = self.model(x)
                        audio_sim = -EU_dist(a, audio_proto)  # B x n_class
                        visual_sim = -EU_dist(v, visual_proto)  # B x n_class
                    elif self.modality_type == 'audio':
                        a, out = self.model.forward_image(x)
                        audio_sim = -EU_dist(a, audio_proto)  # B x n_class
                    elif self.modality_type == 'visual':
                        v, out = self.model.forward_text(x)
                        visual_sim = -EU_dist(v, visual_proto)  # B x n_class
                    else:
                        raise ValueError('No such modality type.')

                    if self.modality_type == 'multimodal':
                        score_a_p = sum([softmax(audio_sim)[i][label[i]] for i in range(audio_sim.size(0))])
                        score_v_p = sum([softmax(visual_sim)[i][label[i]] for i in range(visual_sim.size(0))])
                        ratio_a_p = score_a_p / score_v_p

                        loss_proto_a = criterion(audio_sim, label)
                        loss_proto_v = criterion(visual_sim, label)

                        if ratio_a_p > 1:
                            beta = 0  # audio coef
                            lam = 1 * self.args.MI_alpha  # visual coef
                        elif ratio_a_p < 1:
                            beta = 1 * self.args.MI_alpha
                            lam = 0
                        else:
                            beta = 0
                            lam = 0
                        loss = criterion(out, label) + beta * loss_proto_a + lam * loss_proto_v
                    elif self.modality_type == 'audio':
                        loss_proto_a = criterion(audio_sim, label)
                        loss = criterion(out, label) + self.args.MI_alpha * loss_proto_a
                        # loss.backward()
                        # audio_grads += process_param_grad(self.model, modality='audio')
                    elif self.modality_type == 'visual':
                        loss_proto_v = criterion(visual_sim, label)
                        loss = criterion(out, label) + self.args.MI_alpha * loss_proto_v
                        # loss.backward()
                        # visual_grads += process_param_grad(self.model, modality='visual')
                    loss.backward()
                    optimizer.step()
            else:
                for step, (spec, image, label) in enumerate(train_dataloader):
                    spec = spec.to(self.device)  # B x 257 x 1004(CREMAD 299)
                    image = image.to(self.device)  # B x 1(image count) x 3 x 224 x 224
                    label = label.to(self.device)  # B

                    optimizer.zero_grad()

                    if self.modality_type == 'multimodal':
                        # TODO: make it simpler and easier to extend
                        if self.args.dataset != 'CGMNIST':
                            a, v, out = self.model(spec.unsqueeze(1).float(), image.float())
                        else:
                            a, v, out = self.model(spec, image)  # gray colored
                        audio_sim = -EU_dist(a, audio_proto)  # B x n_class
                        visual_sim = -EU_dist(v, visual_proto)  # B x n_class
                    elif self.modality_type == 'audio':
                        if self.args.dataset != 'CGMNIST':
                            a, out = self.model.forward_audio(spec.unsqueeze(1).float())
                        else:
                            a, out = self.model.forward_gray(spec)
                        audio_sim = -EU_dist(a, audio_proto)  # B x n_class
                    elif self.modality_type == 'visual':
                        if self.args.dataset != 'CGMNIST':
                            v, out = self.model.forward_visual(image.float(), bsz=label.shape[0])
                        else:
                            v, out = self.model.forward_colored(image)
                        visual_sim = -EU_dist(v, visual_proto)  # B x n_class
                    else:
                        raise ValueError('No such modality type.')

                    if self.modality_type == 'multimodal':
                        score_a_p = sum([softmax(audio_sim)[i][label[i]] for i in range(audio_sim.size(0))])
                        score_v_p = sum([softmax(visual_sim)[i][label[i]] for i in range(visual_sim.size(0))])
                        ratio_a_p = score_a_p / score_v_p

                        loss_proto_a = criterion(audio_sim, label)
                        loss_proto_v = criterion(visual_sim, label)

                        if ratio_a_p > 1:
                            beta = 0  # audio coef
                            lam = 1 * self.args.MI_alpha  # visual coef
                        elif ratio_a_p < 1:
                            beta = 1 * self.args.MI_alpha
                            lam = 0
                        else:
                            beta = 0
                            lam = 0

                        # if beta != 0:
                        #     (beta * loss_proto_a).backward(retain_graph=True)
                        #     audio_grads += process_param_grad(self.model, modality='audio')
                        # if lam != 0:
                        #     (lam * loss_proto_v).backward(retain_graph=True)
                        #     visual_grads += process_param_grad(self.model, modality='visual')
                        loss = criterion(out, label) + beta * loss_proto_a + lam * loss_proto_v
                        # loss = criterion(out, label)
                        # loss.backward()
                        # global_grads += process_param_grad(self.model, modality='multi')
                    elif self.modality_type == 'audio':
                        loss_proto_a = criterion(audio_sim, label)
                        loss = criterion(out, label) + self.args.MI_alpha * loss_proto_a
                        # loss.backward()
                        # audio_grads += process_param_grad(self.model, modality='audio')
                    elif self.modality_type == 'visual':
                        loss_proto_v = criterion(visual_sim, label)
                        loss = criterion(out, label) + self.args.MI_alpha * loss_proto_v
                        # loss.backward()
                        # visual_grads += process_param_grad(self.model, modality='visual')
                    loss.backward()
                    optimizer.step()

        if self.args.dataset == 'CREMAD' or self.args.dataset == 'AVE':
            global_params = np.concatenate((process_grad(model.audio_net.parameters()), process_grad(model.visual_net.parameters())), 0)
            local_params = np.concatenate((process_grad(self.model.audio_net.parameters()), process_grad(self.model.visual_net.parameters())), 0)
        elif self.args.dataset == 'CGMNIST':
            global_params = np.concatenate(
                (process_grad(model.gray_net.parameters()), process_grad(model.colored_net.parameters())), 0)
            local_params = np.concatenate(
                (process_grad(self.model.gray_net.parameters()), process_grad(self.model.colored_net.parameters())), 0)
        elif self.args.dataset == 'CrisisMMD':
            global_params = np.concatenate(
                (process_grad(model.imageEncoder.parameters()), process_grad(model.textEncoder.parameters())), 0)
            local_params = np.concatenate(
                (process_grad(self.model.imageEncoder.parameters()), process_grad(self.model.textEncoder.parameters())), 0)
        global_grads = global_params - local_params

        if self.args.dataset == 'CREMAD' or self.args.dataset == 'AVE':
            global_params_a = process_grad(model.audio_net.parameters())
            local_params_a = process_grad(self.model.audio_net.parameters())
            global_params_v = process_grad(model.visual_net.parameters())
            local_params_v = process_grad(self.model.visual_net.parameters())
        elif self.args.dataset == 'CGMNIST':
            global_params_a = process_grad(model.gray_net.parameters())
            local_params_a = process_grad(self.model.gray_net.parameters())
            global_params_v = process_grad(model.colored_net.parameters())
            local_params_v = process_grad(self.model.colored_net.parameters())
        elif self.args.dataset == 'CrisisMMD':
            global_params_a = process_grad(model.imageEncoder.parameters())
            local_params_a = process_grad(self.model.imageEncoder.parameters())
            global_params_v = process_grad(model.textEncoder.parameters())
            local_params_v = process_grad(self.model.textEncoder.parameters())
        audio_grads = global_params_a - local_params_a
        visual_grads = global_params_v - local_params_v

        self.model.load_state_dict(model.state_dict())  # 重置

        # if self.modality_type == 'multimodal':
        #     global_grads[:param_len_a] -= audio_grads
        #     global_grads[param_len_a:] -= visual_grads

        if self.modality_type == 'multimodal':
            acc_a, acc_v, _, _ = evaluate_modality_acc(self.args, self.model, train_dataloader, self.device, epoch=0,
                                                 ratio=1.0, r_proto=True)
            b_ratio = acc_a / acc_v  # modality imbalance ratio
        else:
            b_ratio = 1
        return self.trainset.data_num, global_grads, audio_grads, visual_grads, b_ratio, \
            audio_proto, visual_proto


