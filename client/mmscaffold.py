from typing import Dict, List

import torch
from rich.console import Console

from client.base import ClientBase
import torch.nn as nn
from torch.utils.data import Subset, DataLoader

from config.utils import common_loss
from copy import deepcopy


class SCAFFOLDClient(ClientBase):
    def __init__(
        self,
        args,
        model: torch.nn.Module,
        dataset: str,
        batch_size: int,
        local_epochs: int,
        local_lr: float,
        logger: Console,
        gpu: int,
    ):
        super(SCAFFOLDClient, self).__init__(
            args,
            model,
            dataset,
            batch_size,
            local_epochs,
            local_lr,
            logger,
            gpu,
        )
        self.args = args
        self.c_local_audio: Dict[List[torch.Tensor]] = {}
        self.c_local_visual: Dict[List[torch.Tensor]] = {}
        self.c_local_fusion: Dict[List[torch.Tensor]] = {}
        self.c_diff_audio = []
        self.c_diff_visual = []
        self.c_diff_fusion = []

        self.train_step = 0

    def train(
            self,
            client_id,
            model,
            modality_type,
            c_global,
            evaluate=False,
            verbose=False,
            use_valset=False,
    ):
        c_global_audio, c_global_visual, c_global_fusion = deepcopy(c_global[0]), deepcopy(c_global[1]), deepcopy(c_global[2])

        self.client_id = client_id
        self.modality_type = modality_type
        self.model.load_state_dict(model.state_dict())
        self.get_client_local_dataset()

        if modality_type in ['multimodal', 'audio']:
            if self.client_id not in self.c_local_audio.keys():
                self.c_diff_audio = c_global_audio
            else:
                self.c_diff_audio = []
                for c_l, c_g in zip(self.c_local_audio[self.client_id], c_global_audio):
                    self.c_diff_audio.append(-c_l + c_g)
        else:
            self.c_diff_audio = []
            for c_g in c_global_audio:
                self.c_diff_audio.append(torch.zeros_like(c_g).to(self.device))
        if modality_type in ['multimodal', 'visual']:
            if self.client_id not in self.c_local_visual.keys():
                self.c_diff_visual = c_global_visual
            else:
                self.c_diff_visual = []
                for c_l, c_g in zip(self.c_local_visual[self.client_id], c_global_visual):
                    self.c_diff_visual.append(-c_l + c_g)
        else:
            self.c_diff_visual = []
            for c_g in c_global_visual:
                self.c_diff_visual.append(torch.zeros_like(c_g).to(self.device))
        if self.client_id not in self.c_local_fusion.keys():
            self.c_diff_fusion = c_global_fusion
        else:
            self.c_diff_fusion = []
            for c_l, c_g in zip(self.c_local_fusion[self.client_id], c_global_fusion):
                self.c_diff_fusion.append(-c_l + c_g)

        all_params, data_num_client = self._train()

        # update local control variate
        with torch.no_grad():

            if self.client_id not in self.c_local_audio.keys():
                self.c_local_audio[self.client_id] = [
                    torch.zeros_like(param.data, device=self.device)
                    for param in self.model.audio_net.parameters()
                ]
            if self.client_id not in self.c_local_visual.keys():
                self.c_local_visual[self.client_id] = [
                    torch.zeros_like(param.data, device=self.device)
                    for param in self.model.visual_net.parameters()
                ]
            if self.client_id not in self.c_local_fusion.keys():
                self.c_local_fusion[self.client_id] = [
                    torch.zeros_like(param.data, device=self.device)
                    for param in self.model.fusion_module.parameters()
                ]

            y_delta_audio, y_delta_visual, y_delta_fusion = [], [], []
            c_plus_audio, c_plus_visual, c_plus_fusion = [], [], []
            c_delta_audio, c_delta_visual, c_delta_fusion = [], [], []

            # compute y_delta (difference of model before and after training)
            for param_l, param_g in zip(self.model.audio_net.parameters(), model.audio_net.parameters()):
                y_delta_audio.append(param_g.data - param_l.data)
            for param_l, param_g in zip(self.model.visual_net.parameters(), model.visual_net.parameters()):
                y_delta_visual.append(param_g.data - param_l.data)  # 判断对于单模态client，是否确实另一个模态的参数没有被更新
            if self.args.fusion_method == 'concat':
                for param_l, param_g in zip(self.model.fusion_module.parameters(), model.fusion_module.parameters()):
                    y_delta_fusion.append(param_g.data - param_l.data)

            # compute c_plus
            coef = 1 / (self.train_step * self.local_lr)  # 为每个client根据数据的多少单独计算这个值
            if self.modality_type == 'multimodal':
                for c_l, c_g, diff in zip(self.c_local_audio[self.client_id], c_global_audio, y_delta_audio):
                    c_plus_audio.append(c_l - c_g + coef * diff)  # lingyizhong
                for c_l, c_g, diff in zip(self.c_local_visual[self.client_id], c_global_visual, y_delta_visual):
                    c_plus_visual.append(c_l - c_g + coef * diff)
                for c_l, c_g, diff in zip(self.c_local_fusion[self.client_id], c_global_fusion, y_delta_fusion):
                    c_plus_fusion.append(c_l - c_g + coef * diff)
            elif self.modality_type == 'audio':
                for c_l, c_g, diff in zip(self.c_local_audio[self.client_id], c_global_audio, y_delta_audio):
                    c_plus_audio.append(c_l - c_g + coef * diff)
                for c_l, c_g, diff in zip(self.c_local_fusion[self.client_id], c_global_fusion, y_delta_fusion):
                    c_plus_fusion.append(c_l - c_g + coef * diff)
            elif self.modality_type == 'visual':
                for c_l, c_g, diff in zip(self.c_local_visual[self.client_id], c_global_visual, y_delta_visual):
                    c_plus_visual.append(c_l - c_g + coef * diff)
                for c_l, c_g, diff in zip(self.c_local_fusion[self.client_id], c_global_fusion, y_delta_fusion):
                    c_plus_fusion.append(c_l - c_g + coef * diff)
            else:
                raise ValueError('error.')

            # compute c_delta
            if self.modality_type == 'multimodal':
                for c_p, c_l in zip(c_plus_audio, self.c_local_audio[self.client_id]):
                    c_delta_audio.append(c_p.data - c_l.data)  # 对于缺失此模态的client上，这个值应该为0
                for c_p, c_l in zip(c_plus_visual, self.c_local_visual[self.client_id]):
                    c_delta_visual.append(c_p.data - c_l.data)
                for c_p, c_l in zip(c_plus_fusion, self.c_local_fusion[self.client_id]):
                    c_delta_fusion.append(c_p.data - c_l.data)
            elif self.modality_type == 'audio':
                for c_p, c_l in zip(c_plus_audio, self.c_local_audio[self.client_id]):
                    c_delta_audio.append(c_p.data - c_l.data)  # 对于缺失此模态的client上，这个值应该为0
                for c_l in self.c_local_visual[self.client_id]:
                    c_delta_visual.append(torch.zeros_like(c_l, device=self.device))
                for c_p, c_l in zip(c_plus_fusion, self.c_local_fusion[self.client_id]):
                    c_delta_fusion.append(c_p.data - c_l.data)
            elif self.modality_type == 'visual':
                for c_l in self.c_local_audio[self.client_id]:
                    c_delta_audio.append(torch.zeros_like(c_l, device=self.device))  # 对于缺失此模态的client上，这个值应该为0
                for c_p, c_l in zip(c_plus_visual, self.c_local_visual[self.client_id]):
                    c_delta_visual.append(c_p.data - c_l.data)
                for c_p, c_l in zip(c_plus_fusion, self.c_local_fusion[self.client_id]):
                    c_delta_fusion.append(c_p.data - c_l.data)

            if self.modality_type in ['multimodal', 'audio']:
                # 单模态client不更新另一个模态的c
                self.c_local_audio[self.client_id] = c_plus_audio
            if self.modality_type in ['multimodal', 'visual']:
                self.c_local_visual[self.client_id] = c_plus_visual
            self.c_local_fusion[self.client_id] = c_plus_fusion

        return all_params, [c_delta_audio, c_delta_visual, c_delta_fusion], data_num_client

    def _train(self):
        self.train_step = 0

        optimizer: torch.optim.Optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.args.local_lr, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.args.lr_decay_step, self.args.lr_decay_ratio)

        criterion = nn.CrossEntropyLoss()

        train_dataloader = DataLoader(self.trainset, batch_size=self.batch_size,
                                      shuffle=True, pin_memory=False)  # 计算机的内存充足的时候，可以设置pin_memory=True

        for le in range(self.local_epochs):
            self.model.train()
            _loss = 0
            for step, (spec, image, label) in enumerate(train_dataloader):
                self.train_step += 1

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

                loss = criterion(out, label) + loss_common
                _loss += loss.item()
                loss.backward()

                if self.modality_type == 'multimodal':
                    for param, c_d in zip(self.model.audio_net.parameters(), self.c_diff_audio):
                        param.grad += c_d.data
                    for param, c_d in zip(self.model.visual_net.parameters(), self.c_diff_visual):
                        param.grad += c_d.data
                    for param, c_d in zip(self.model.fusion_module.parameters(), self.c_diff_fusion):
                        param.grad += c_d.data
                elif self.modality_type == 'audio':
                    for param, c_d in zip(self.model.audio_net.parameters(), self.c_diff_audio):
                        param.grad += c_d.data
                    for param, c_d in zip(self.model.fusion_module.parameters(), self.c_diff_fusion):
                        param.grad += c_d.data
                elif self.modality_type == 'visual':
                    for param, c_d in zip(self.model.visual_net.parameters(), self.c_diff_visual):
                        param.grad += c_d.data
                    for param, c_d in zip(self.model.fusion_module.parameters(), self.c_diff_fusion):
                        param.grad += c_d.data
                else:
                    raise ValueError('No such modality type.')

                optimizer.step()
            # print('diff: ', self.c_diff_audio[0][0][0][0])
            if self.args.optimizer == 'SGD':
                scheduler.step()

        print('client {} loss: '.format(self.client_id), _loss / len(train_dataloader))

        model_params = [self.model.audio_net.state_dict(), self.model.visual_net.state_dict(),
                        self.model.fusion_module.state_dict()]
        return model_params, self.trainset.data_num

