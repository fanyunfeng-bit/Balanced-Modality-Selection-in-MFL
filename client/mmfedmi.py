# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 2023
fedavg with modality imbalance technique (client)

local CE + local PMR + local cross-modal relation distillation

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
from data.utils.util import get_train_dataset, get_val_dataset, CremadDataset
from models.basic_model import AVClassifier
from torch.utils.data import Subset, DataLoader
from copy import deepcopy
from typing import Dict, List, OrderedDict, Tuple
import numpy as np
import torch.nn as nn
from client.base import ClientBase


class FedMIClient(ClientBase):
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
        super(FedMIClient, self).__init__(
            args,
            model,
            dataset,
            batch_size,
            local_epochs,
            local_lr,
            logger,
            gpu,
        )

        self.coef1 = 1.0  # local PMR
        self.coef2 = 1.0  # cross modal relation distillation
        self.coef3 = 1.0  #

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
        # self.lr = self.args.local_lr - (self.args.local_lr - self.args.lr_min) / self.args.global_epochs * global_epoch
        self.lr = self.args.local_lr
        self.client_id = client_id
        self.modality_type = modality_type
        self.model.load_state_dict(model.state_dict())
        self.get_client_local_dataset()
        if self.client_id not in self.visited_client:
            self.visited_client.append(self.client_id)

        self.global_proto = global_proto

        all_params, data_num_client = self._train()

        return all_params, data_num_client

    def _train(self):
        optimizer: torch.optim.Optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.args.lr_decay_step, self.args.lr_decay_ratio)

        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)

        train_dataloader = DataLoader(self.trainset, batch_size=self.batch_size,
                                      shuffle=True, pin_memory=False)  # 计算机的内存充足的时候，可以设置pin_memory=True

        # if (self.args.MI_correct or self.args.relation_distill) and self.args.modulation_starts <= self.global_epoch <= self.args.modulation_ends:
        #     self.audio_proto[self.client_id], self.visual_proto[self.client_id] = calculate_prototype(self.args,
        #                                                                                               self.model,
        #                                                                                               train_dataloader,
        #                                                                                               self.device,
        #                                                                                               epoch=0)
        for le in range(self.local_epochs):

            self.model.train()
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
                        a, v, out = self.model(spec.unsqueeze(1).float(), image.float())
                    else:
                        a, v, out = self.model(spec, image)  # gray colored
                    if self.args.local_align:  # perform modality alignment with contrastive learning
                        loss_common = common_loss(a, v)
                elif self.modality_type == 'audio':
                    out = self.model.forward_audio(spec.unsqueeze(1).float())
                elif self.modality_type == 'visual':
                    out = self.model.forward_visual(image.float(), bsz=label.shape[0])
                else:
                    raise ValueError('No such modality type.')

                loss = criterion(out, label)

                if (self.args.MI_correct or self.args.relation_distill) and self.global_proto is not None and \
                        self.args.modulation_starts <= self.global_epoch <= self.args.modulation_ends and self.modality_type == 'multimodal':
                    audio_sim = -EU_dist(a, self.global_proto[0])  # B x n_class
                    visual_sim = -EU_dist(v, self.global_proto[1])  # B x n_class
                    prob_audio = softmax(audio_sim)
                    prob_visual = softmax(visual_sim)

                    # 根据当前batch算的每个模态的优劣，是否改成根据累计见过的样本进行计算
                    score_a_p = sum([prob_audio[i][label[i]] for i in range(audio_sim.size(0))])
                    score_v_p = sum([prob_visual[i][label[i]] for i in range(visual_sim.size(0))])
                    ratio_a_p = score_a_p / score_v_p
                    if self.args.MI_correct:
                        # loss with local proto
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
                        loss += self.coef1 * (beta * loss_proto_a + lam * loss_proto_v) + loss_common

                    if self.args.relation_distill:
                        # audio_probs_c = [0 for _ in range(self.n_classes)]
                        # visual_probs_c = [0 for _ in range(self.n_classes)]
                        # for i in range(audio_sim.size(0)):
                        #     audio_probs_c[label[i]] += prob_audio[i][label[i]]
                        #     visual_probs_c[label[i]] += prob_visual[i][label[i]]
                        #
                        if ratio_a_p > 1:
                        #     class_relation = []
                        #     for nc in range(self.n_classes):
                        #         if audio_probs_c[nc] > visual_probs_c[nc]:  # 被严重抑制的标准是否要改
                        #             class_relation.append(nc)
                            loss_cmr = relation_loss(a, v, self.audio_proto[self.client_id], self.visual_proto[self.client_id], label, self.n_classes, self.args.temp, a_detach=True)
                        else:
                            # class_relation = []
                            # for nc in range(self.n_classes):
                            #     if audio_probs_c[nc] < visual_probs_c[nc]:  # 被严重抑制的标准是否要改
                            #         class_relation.append(nc)
                            loss_cmr = relation_loss(a, v, self.audio_proto[self.client_id], self.visual_proto[self.client_id],
                                          label, self.n_classes, self.args.temp, a_detach=False)
                        loss += self.coef2 * loss_cmr
                        # print(loss_cmr)
                else:
                    loss += loss_common

                _loss += loss.item()
                loss.backward()
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
        # train uni classifier
        if (self.global_epoch+1) % 10 == 0:
            train_visual = True
            train_audio = True

            train_dataset = CremadDataset(self.args, mode='train', fl=False)
            c_train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size,
                                            shuffle=True, pin_memory=False)  # 计算机的内存充足的时候，可以设置pin_memory=True

            val_dataset = get_val_dataset(self.args, self.args.dataset)
            val_dataloader = DataLoader(val_dataset, batch_size=self.args.val_batch_size, shuffle=False,
                                        pin_memory=False)
            print(self.count_class)
            if train_visual:
                classifier = deepcopy(self.model.fusion_module.fc_out)  # 1024
                encoder = self.model.visual_net
                self.args.train_modality = 'visual'

                class_optimizer: torch.optim.Optimizer = torch.optim.SGD(
                    classifier.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
                class_scheduler = torch.optim.lr_scheduler.StepLR(class_optimizer, self.args.lr_decay_step,
                                                                  self.args.lr_decay_ratio)

                batch_loss = train_uniclassifier_epoch(self.args, self.global_epoch, encoder, classifier, self.device,
                                                       c_train_dataloader, class_optimizer, class_scheduler)
                acc, acc_class = valid_uniclassifier(self.args, encoder, classifier, self.device, val_dataloader)
                print('visual_acc: ', acc, acc_class)

            if train_audio:
                classifier = deepcopy(self.model.fusion_module.fc_out)  # 1024
                encoder = self.model.audio_net
                self.args.train_modality = 'audio'

                class_optimizer: torch.optim.Optimizer = torch.optim.SGD(
                    classifier.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
                class_scheduler = torch.optim.lr_scheduler.StepLR(class_optimizer, self.args.lr_decay_step,
                                                                  self.args.lr_decay_ratio)

                batch_loss = train_uniclassifier_epoch(self.args, self.global_epoch, encoder, classifier, self.device,
                                                       c_train_dataloader, class_optimizer, class_scheduler)
                acc, acc_class = valid_uniclassifier(self.args, encoder, classifier, self.device, val_dataloader)
                print('audio_acc: ', acc, acc_class)

        print('client {} loss: '.format(self.client_id), _loss / len(train_dataloader))

        model_params = [self.model.audio_net.state_dict(), self.model.visual_net.state_dict(),
                        self.model.fusion_module.state_dict()]
        return model_params, self.trainset.data_num



