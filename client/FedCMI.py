# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 2023
fedavg with cross modal distillation (client)

@author: fyf
"""

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

from config.utils import fix_random_seed, common_loss, calculate_prototype, EU_dist, relation_loss
from config.utils import trainable_params
from data.utils.util import get_train_dataset, get_val_dataset, CremadDataset
from models.basic_model import AVClassifier
from torch.utils.data import Subset, DataLoader
from copy import deepcopy
from typing import Dict, List, OrderedDict, Tuple
import numpy as np
import torch.nn as nn
from client.base import ClientBase

import torch.nn.functional as F


class FedCMIClient(ClientBase):
    def __init__(
            self,
            args,
            model: torch.nn.Module,
            dataset: str,
            batch_size: int,
            local_epochs: int,
            local_lr: float,
            logger: Console,
            gpu: int, ):
        super(FedCMIClient, self).__init__(
            args,
            model,
            dataset,
            batch_size,
            local_epochs,
            local_lr,
            logger,
            gpu,
        )

        self.coef1 = 2.0  # local PMR
        self.coef2 = self.args.MI_alpha  # cross modal distillation
        self.coef3 = 2.0  #

        self.client_av_branch = {i: self.model.audio_v_branch.state_dict() for i in range(self.args.client_num)}
        self.client_va_branch = {i: self.model.visual_a_branch.state_dict() for i in range(self.args.client_num)}
        self.client_av_classifier = {i: self.model.av_classifier.state_dict() for i in range(self.args.client_num)}
        self.client_va_classifier = {i: self.model.va_classifier.state_dict() for i in range(self.args.client_num)}

        self.last_round_ratio = 0

    def train(
            self,
            client_id: int,
            model,
            modality_type=None,
            evaluate=True,
            global_epoch=0,
            use_valset=True,
            global_proto=None
    ):
        self.global_epoch = global_epoch
        self.global_model = deepcopy(model)

        if self.global_epoch > 80:
            self.lr = self.args.local_lr - (self.args.local_lr - self.args.lr_min) / self.args.global_epochs * global_epoch
        else:
            self.lr = self.args.local_lr

        self.client_id = client_id
        self.modality_type = modality_type

        if self.args.personalized_branch:
            self.model.load_state_dict(model.state_dict())
            self.model.audio_v_branch.load_state_dict(self.client_av_branch[client_id])
            self.model.visual_a_branch.load_state_dict(self.client_va_branch[client_id])
            self.model.av_classifier.load_state_dict(self.client_av_classifier[client_id])
            self.model.va_classifier.load_state_dict(self.client_va_classifier[client_id])
        else:
            self.model.load_state_dict(model.state_dict())
        self.get_client_local_dataset()
        if self.client_id not in self.visited_client:
            self.visited_client.append(self.client_id)

        self.global_proto = global_proto

        all_params, data_num_client = self._train()

        return all_params, data_num_client

    def _train(self):

        # global_params = [p.clone().detach() for p in trainable_params(self.global_model)]

        optimizer: torch.optim.Optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.args.lr_decay_step, self.args.lr_decay_ratio)

        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)

        train_dataloader = DataLoader(self.trainset, batch_size=self.batch_size,
                                      shuffle=True, pin_memory=False)  # 计算机的内存充足的时候，可以设置pin_memory=True

        # 判断不同class的好坏，得到不同class的average ground truth probs
        if self.modality_type == 'multimodal' and self.global_epoch >= self.args.warmup_epoch:
            probs_per_class_audio, probs_per_class_visual = self.get_average_probs(train_dataloader)
            average = 0
            zero_class = 0
            for c in range(self.n_classes):
                if probs_per_class_visual[c] == 0:
                    zero_class += 1
                else:
                    average += probs_per_class_audio[c] / probs_per_class_visual[c]
            average /= (self.n_classes - zero_class)

            class_wise_dw = [1 for _ in range(self.n_classes)]
            class_wise_temp = [1 for _ in range(self.n_classes)]
            if average > 1:  # audio dominant
                for c in range(self.n_classes):
                    if probs_per_class_visual[c] == 0:
                        pass
                    else:
                        # class_wise_dw[c] = 1+self.coef3*torch.log(probs_per_class_audio[c] / probs_per_class_visual[c] / average) if probs_per_class_audio[c] / probs_per_class_visual[c] > average else 1
                        class_wise_temp[c] = 1/(1+self.coef3*torch.log(probs_per_class_audio[c] / probs_per_class_visual[c] / average)) if probs_per_class_audio[c] / probs_per_class_visual[c] > average else 1
            else:
                for c in range(self.n_classes):
                    if probs_per_class_visual[c] == 0:
                        pass
                    else:
                        # class_wise_dw[c] = 1+self.coef3*torch.log(probs_per_class_visual[c] / probs_per_class_audio[c] / (1/average)) if probs_per_class_visual[c] / probs_per_class_audio[c] > 1/average else 1
                        class_wise_temp[c] = 1/(1 + self.coef3 * torch.log(probs_per_class_visual[c] / probs_per_class_audio[c] / (1 / average))) if probs_per_class_visual[c] / probs_per_class_audio[c] > 1 / average else 1
            # print(class_wise_dw, class_wise_temp)

        for le in range(self.local_epochs):

            self.model.train()
            self.global_model.eval()
            _loss = 0
            for step, (spec, image, label) in enumerate(train_dataloader):
                spec = spec.to(self.device)  # B x 257 x 1004(CREMAD 299)
                image = image.to(self.device)  # B x 1(image count) x 3 x 224 x 224
                label = label.to(self.device)  # B

                optimizer.zero_grad()

                loss_common = 0
                if self.modality_type == 'multimodal':
                    # TODO: make it simpler and easier to extend
                    if self.args.dataset != 'CGMNIST':
                        aa, av, va, vv, out, av_out, va_out = self.model(spec.unsqueeze(1).float(), image.float())
                        with torch.no_grad():
                            _, _, _, _, g_out, g_av_out, g_va_out = self.global_model(spec.unsqueeze(1).float(), image.float())
                    else:
                        a, v, out = self.model(spec, image)  # gray colored

                    # xx做CE loss相当于在促进当前模态，是否要改成只有被抑制模态才有这个loss
                    loss = criterion(out, label) + self.coef1*criterion(av_out[:label.shape[0]], label) + self.coef1*criterion(va_out[label.shape[0]:], label)
                elif self.modality_type == 'audio':
                    _, _, out, av_out = self.model.forward_audio(spec.unsqueeze(1).float())
                    loss = criterion(out, label)
                elif self.modality_type == 'visual':
                    _, _, out, va_out = self.model.forward_visual(image.float(), bsz=label.shape[0])
                    loss = criterion(out, label)
                else:
                    raise ValueError('No such modality type.')

                loss_rd = 0
                if self.args.cross_modal_distill and self.modality_type == 'multimodal' and self.global_epoch >= self.args.warmup_epoch:
                    a_probs = softmax(av_out[:label.shape[0]])
                    v_probs = softmax(va_out[label.shape[0]:])
                    score_a = sum(a_probs[i][label[i]] for i in range(label.shape[0]))
                    score_v = sum(v_probs[i][label[i]] for i in range(label.shape[0]))
                    ratio = score_a / score_v
                    # print(ratio)
                    if ratio > 1:
                        # loss_fd = torch.sum((aa.detach() - va) * (aa.detach() - va), dim=1).mean()  # feature distill

                        if self.global_epoch >= self.args.warmup_epoch:
                            if self.args.class_wise_w:
                                loss_rd = 0
                                p_a, p_v = F.softmax(av_out / self.args.temp, dim=1), F.softmax(va_out/self.args.temp, dim=1)
                                for ll in range(label.shape[0]):
                                    loss_rd += -(class_wise_dw[label[ll]] * torch.sum(p_a[ll].detach() * torch.log(p_v[ll])))
                                loss_rd /= label.shape[0]
                            if self.args.class_wise_t:
                                p_a, p_v = torch.zeros_like(av_out).to(self.device), torch.zeros_like(va_out).to(
                                    self.device)
                                for ll in range(label.shape[0]):
                                    p_a[ll] = F.softmax(g_av_out[ll] / class_wise_temp[label[ll]], dim=0)
                                    # p_a[label.shape[0] + ll] = F.softmax(
                                    #     av_out[label.shape[0] + ll] / class_wise_temp[label[ll]], dim=0)
                                    p_v[ll] = F.softmax(va_out[ll] / class_wise_temp[label[ll]], dim=0)
                                    # p_v[label.shape[0] + ll] = F.softmax(
                                    #     va_out[label.shape[0] + ll] / class_wise_temp[label[ll]], dim=0)
                                loss_rd = -torch.sum(p_a[:label.shape[0]].detach() * torch.log(p_v[:label.shape[0]]),
                                                     dim=1).mean()
                        else:
                            p_a, p_v = F.softmax(g_av_out / self.args.temp, dim=1), F.softmax(va_out/self.args.temp, dim=1)
                            loss_rd = -torch.sum(p_a[:label.shape[0]].detach() * torch.log(p_v[:label.shape[0]]), dim=1).mean()
                            # loss_rd = F.kl_div(p_v[:label.shape[0]].log(), p_a[:label.shape[0]].detach(), reduction='mean')
                    else:
                        # loss_fd = torch.sum((vv.detach() - av) * (vv.detach() - av), dim=1).mean()  # feature distill

                        if self.global_epoch >= self.args.warmup_epoch:
                            if self.args.class_wise_w:
                                loss_rd = 0
                                p_a, p_v = F.softmax(av_out / self.args.temp, dim=1), F.softmax(va_out / self.args.temp, dim=1)
                                for ll in range(label.shape[0]):
                                    loss_rd += -(class_wise_dw[label[ll]] * torch.sum(p_v[label.shape[0]+ll].detach() * torch.log(p_a[label.shape[0]+ll])))
                                loss_rd /= label.shape[0]
                            if self.args.class_wise_t:
                                p_a, p_v = torch.zeros_like(av_out).to(self.device), torch.zeros_like(va_out).to(
                                    self.device)
                                for ll in range(label.shape[0]):
                                    # p_a[ll] = F.softmax(av_out[ll] / class_wise_temp[label[ll]], dim=0)
                                    p_a[label.shape[0] + ll] = F.softmax(
                                        av_out[label.shape[0] + ll] / class_wise_temp[label[ll]], dim=0)
                                    # p_v[ll] = F.softmax(va_out[ll] / class_wise_temp[label[ll]], dim=0)
                                    p_v[label.shape[0] + ll] = F.softmax(
                                        g_va_out[label.shape[0] + ll] / class_wise_temp[label[ll]], dim=0)
                                loss_rd = -torch.sum(p_v[label.shape[0]:].detach() * torch.log(p_a[label.shape[0]:]),
                                                     dim=1).mean()
                        else:
                            p_a, p_v = F.softmax(av_out / self.args.temp, dim=1), F.softmax(g_va_out/self.args.temp, dim=1)
                            loss_rd = - torch.sum(p_v[label.shape[0]:].detach() * torch.log(p_a[label.shape[0]:]),
                                                  dim=1).mean()
                            # loss_rd = F.kl_div(p_a[label.shape[0]:].log(), p_v[label.shape[0]:].detach(), reduction='mean')
                    # print('loss_rd: ', loss_rd)
                    loss += self.coef2 * loss_rd

                    self.last_round_ratio = ratio

                _loss += loss.item()
                loss.backward()

                if self.args.prox:
                    for w, w_t in zip(trainable_params(self.model), trainable_params(self.global_model)):
                        if w.grad is not None:
                            w.grad.data += self.args.mu * (w.data - w_t.data)

                optimizer.step()

            if self.args.optimizer == 'SGD':
                scheduler.step()

            # if le % 10 == 0:
            #     acc = self.validate()
            #     print('acc in client {}: '.format(self.client_id), acc)

        self.audio_proto[self.client_id], self.visual_proto[self.client_id] = calculate_prototype(self.args,
                                                                                                  self.model,
                                                                                                  train_dataloader,
                                                                                                  self.device,
                                                                                                  epoch=0)

        self.client_av_branch[self.client_id] = self.model.audio_v_branch.state_dict()
        self.client_va_branch[self.client_id] = self.model.visual_a_branch.state_dict()
        self.client_av_classifier[self.client_id] = self.model.av_classifier.state_dict()
        self.client_va_classifier[self.client_id] = self.model.va_classifier.state_dict()

        print('client {} loss: '.format(self.client_id), _loss / len(train_dataloader))

        model_params = [self.model.audio_net.state_dict(), self.model.visual_net.state_dict(), self.model.audio_a_branch.state_dict(),
                        self.model.audio_v_branch.state_dict(), self.model.visual_a_branch.state_dict(),
                        self.model.visual_v_branch.state_dict(), self.model.fusion_module.state_dict(),
                        self.model.av_classifier.state_dict(), self.model.va_classifier.state_dict()]
        return model_params, self.trainset.data_num

    def get_average_probs(self, dataloader):
        softmax = nn.Softmax(dim=1)
        self.model.eval()
        probs_per_class_audio = [0 for _ in range(self.n_classes)]
        probs_per_class_visual = [0 for _ in range(self.n_classes)]

        count_class = [0 for _ in range(self.n_classes)]

        with torch.no_grad():
            for step, (spec, image, label) in enumerate(dataloader):
                spec = spec.to(self.device)  # B x 257 x 1004(CREMAD 299)
                image = image.to(self.device)  # B x 1(image count) x 3 x 224 x 224
                label = label.to(self.device)  # B

                if self.args.dataset != 'CGMNIST':
                    aa, av, va, vv, out, av_out, va_out = self.model(spec.unsqueeze(1).float(), image.float())
                else:
                    a, v, out = self.model(spec, image)  # gray colored

                av_probs, va_probs = softmax(av_out[:label.shape[0]]), softmax(va_out[label.shape[0]:])
                for i in range(label.shape[0]):
                    probs_per_class_audio[label[i]] += av_probs[i, label[i]].data
                    probs_per_class_visual[label[i]] += va_probs[i, label[i]].data
                    count_class[label[i]] += 1

            for c in range(self.n_classes):
                if count_class[c] == 0:
                    pass
                else:
                    probs_per_class_audio[c] /= count_class[c]
                    probs_per_class_visual[c] /= count_class[c]

        return probs_per_class_audio, probs_per_class_visual





