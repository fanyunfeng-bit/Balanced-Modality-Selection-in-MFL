# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 20:15:48 2023

@author: Stu
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 2023
fedavg with cross modal distillation (server)

@author: fyf
"""

import os
import pickle
import random

import torch
from path import Path
from rich.console import Console
from rich.progress import track
from tqdm import tqdm

from client.FedCMI import FedCMIClient
from config.utils import fix_random_seed, allocate_client_modality, EU_dist, dot_product_angle_tensor, \
    grad_amplitude_diff
from data.utils.util import get_val_dataset
from models.basic_model import AVClassifier
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from copy import deepcopy
from server.base import ServerBase
from config.utils import get_args


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class FedCMIServer(ServerBase):
    def __init__(self):
        super(FedCMIServer, self).__init__(get_args(), "FedCMI")
        fix_random_seed(self.args.seed)

        self.trainer = FedCMIClient(args=self.args,
                                    model=deepcopy(self.model),
                                    dataset=self.args.dataset,
                                    batch_size=self.args.batch_size,
                                    local_epochs=self.args.local_epochs,
                                    local_lr=self.args.local_lr,
                                    logger=self.logger,
                                    gpu=self.args.gpu, )

        self.global_audio_proto = []
        self.global_visual_proto = []

        self.this_file_dir = Path(__file__).parent.parent.abspath()
        self.save_dir = self.this_file_dir / 'checkpoints' / self.algo / self.args.dataset / 'client-{}-cr-{}'. \
                            format(self.args.client_num, self.args.client_num_per_round) / 'alpha-{}-mmratio-{}' \
                            .format(self.args.alpha, self.args.multi_ratio) / 'local-epoch-{}'.format(
            self.args.local_epochs) / 'lr-{}'. \
                            format(self.args.local_lr) / 'CMI-{}'.format(str(self.args.cross_modal_distill)) / 'Mutual-Distill-False' / str(self.args.branch_type) + '-4layer-cwt-' + str(self.args.class_wise_t) + '/' + str(self.args.MI_alpha) + '/prox-' + str(self.args.prox) + '/' + str(self.args.fusion_method) # cwd: class-wise distillation

        self.log_dir = self.this_file_dir / 'logs' / self.algo / self.args.dataset / 'client-{}-cr-{}'. \
                           format(self.args.client_num, self.args.client_num_per_round) / 'alpha-{}-mmratio-{}' \
                           .format(self.args.alpha, self.args.multi_ratio) / 'local-epoch-{}'.format(
            self.args.local_epochs) / 'lr-{}'. \
                           format(self.args.local_lr) / 'CMI-{}'.format(str(self.args.cross_modal_distill)) /'Mutual-Distill-False' / str(self.args.branch_type) + '-4layer-cwt-' + str(self.args.class_wise_t) + '/' + str(self.args.MI_alpha) + '/prox-' + str(self.args.prox) + '/' + str(self.args.fusion_method)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.log_file = self.log_dir + '/log.txt'
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

    def train(self):
        print('start training...')
        progress_bar = tqdm(range(self.global_epochs), "Training...")

        f_log = open(self.log_file, 'a')

        for E in progress_bar:
            selected_clients = random.sample(
                self.client_id_indices, self.args.client_num_per_round
            )

            all_audio_params = []
            all_visual_params = []
            all_fusion_params = []
            all_audio_a_params = []
            all_visual_v_params = []
            all_audio_v_params = []
            all_visual_a_params = []
            all_av_classifier_params = []
            all_va_classifier_params = []
            all_client_data_num = []

            for client_id in selected_clients:
                print('client {} starts training...'.format(client_id))
                if client_id in self.mm_client:
                    modality_type = 'multimodal'
                elif client_id in self.audio_client:
                    modality_type = 'audio'
                elif client_id in self.visual_client:
                    modality_type = 'visual'
                else:
                    raise ValueError('Non-exist modality type')

                if E == 0:
                    global_proto = None
                else:
                    global_proto = None
                all_params, data_num_client = self.trainer.train(
                    client_id=client_id,
                    model=deepcopy(self.model),
                    modality_type=modality_type,
                    global_epoch=E,
                    global_proto=global_proto
                )

                all_audio_params.append(deepcopy(all_params[0]))
                all_visual_params.append(deepcopy(all_params[1]))
                all_audio_a_params.append(deepcopy(all_params[2]))
                all_audio_v_params.append(deepcopy(all_params[3]))
                all_visual_a_params.append(deepcopy(all_params[4]))
                all_visual_v_params.append(deepcopy(all_params[5]))
                all_fusion_params.append(deepcopy(all_params[6]))
                all_av_classifier_params.append(deepcopy(all_params[7]))
                all_va_classifier_params.append(deepcopy(all_params[8]))
                all_client_data_num.append(data_num_client)

            # # cal global prototypes
            # audio_prototypes = torch.zeros(self.n_classes, self.args.embed_dim).to(self.device)
            # visual_prototypes = torch.zeros(self.n_classes, self.args.embed_dim).to(self.device)
            # data_nums = 0
            # for client in selected_clients:
            #     a_proto, v_proto, data_num = self.trainer.get_local_prototype(client, self.model)
            #     audio_prototypes += a_proto * data_num
            #     visual_prototypes += v_proto * data_num
            #     data_nums += data_num
            # audio_prototypes /= data_nums
            # visual_prototypes /= data_nums
            # # 可以是momentum更新
            # self.global_audio_proto = audio_prototypes
            # self.global_visual_proto = visual_prototypes

            # aggregation
            global_audio_weights, global_visual_weights, global_audio_a_weights, global_audio_v_weights, \
                global_visual_a_weights, global_visual_v_weights, global_fusion_weights, global_av_class_weights, \
                global_va_class_weights, audio_update, visual_update = \
                self.aggregate([all_audio_params, all_visual_params, all_audio_a_params, all_audio_v_params, all_visual_a_params,
                                all_visual_v_params, all_fusion_params, all_av_classifier_params, all_va_classifier_params],
                               all_client_data_num, selected_clients)
            if audio_update:
                self.model.audio_net.load_state_dict(global_audio_weights)
                self.model.audio_a_branch.load_state_dict(global_audio_a_weights)
                self.model.audio_v_branch.load_state_dict(global_audio_v_weights)
                self.model.av_classifier.load_state_dict(global_av_class_weights)
            if visual_update:
                self.model.visual_net.load_state_dict(global_visual_weights)
                self.model.visual_a_branch.load_state_dict(global_visual_a_weights)
                self.model.visual_v_branch.load_state_dict(global_visual_v_weights)
                self.model.va_classifier.load_state_dict(global_va_class_weights)
            self.model.fusion_module.load_state_dict(global_fusion_weights)

            acc, acc_av, acc_va, acc_a, acc_v = self.validate()
            print('accuracy for round {}: '.format(E), acc, acc_av, acc_va, acc_a, acc_v)

            f_log.write(str(E) +
                        "\t" + str(acc) +
                        "\t" + str(acc_av) +
                        "\t" + str(acc_va) +
                        "\t" + str(acc_a) +
                        "\t" + str(acc_v) +
                        "\n")
            f_log.flush()

            # if E % self.args.save_period == 0:
            #     torch.save(
            #         self.model.audio_net.state_dict(),
            #         self.save_dir / "global_model_audio.pt",
            #     )
            #     torch.save(
            #         self.model.visual_net.state_dict(),
            #         self.save_dir / "global_model_visual.pt",
            #     )
            #     torch.save(
            #         self.model.audio_a_branch.state_dict(),
            #         self.save_dir / "global_model_audio_a.pt",
            #     )
            #     torch.save(
            #         self.model.visual_v_branch.state_dict(),
            #         self.save_dir / "global_model_visual_v.pt",
            #     )
            #     torch.save(
            #         self.model.fusion_module.state_dict(),
            #         self.save_dir / "global_model_fusion.pt",
            #     )
            #     with open(self.save_dir / "epoch.pkl", "wb") as f:
            #         pickle.dump(E, f)
        f_log.close()

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
        acc_av = [0.0 for _ in range(n_classes)]
        acc_va = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        with torch.no_grad():
            for step, (spec, image, label) in enumerate(val_dataloader):
                spec = spec.to(self.device)
                image = image.to(self.device)
                label = label.to(self.device)

                if self.args.visual_only:
                    va, vv, out, va_out = self.model.forward_visual(image.float(), bsz=label.shape[0])
                elif self.args.audio_only:
                    aa, av, out, av_out = self.model.forward_audio(spec.unsqueeze(1).float())
                else:
                    if self.args.dataset != 'CGMNIST':
                        aa, av, va, vv, out, av_out, va_out = self.model(spec.unsqueeze(1).float(), image.float())

                prediction = softmax(out)
                other_modality = torch.zeros_like(aa).to(aa.device)
                _, _, logit_av = self.model.fusion_module(aa, av)
                _, _, logit_va = self.model.fusion_module(va, vv)
                _, _, logit_a = self.model.fusion_module(aa, other_modality)
                _, _, logit_v = self.model.fusion_module(other_modality, vv)
                prediction_av = softmax(logit_av)
                prediction_va = softmax(logit_va)
                prediction_a = softmax(logit_a)
                prediction_v = softmax(logit_v)

                for i in range(image.shape[0]):

                    ma = np.argmax(prediction[i].cpu().data.numpy())
                    av = np.argmax(prediction_av[i].cpu().data.numpy())
                    va = np.argmax(prediction_va[i].cpu().data.numpy())
                    a = np.argmax(prediction_a[i].cpu().data.numpy())
                    v = np.argmax(prediction_v[i].cpu().data.numpy())
                    num[label[i]] += 1.0

                    # pdb.set_trace()
                    if np.asarray(label[i].cpu()) == ma:
                        acc[label[i]] += 1.0
                    if np.asarray(label[i].cpu()) == av:
                        acc_av[label[i]] += 1.0
                    if np.asarray(label[i].cpu()) == va:
                        acc_va[label[i]] += 1.0
                    if np.asarray(label[i].cpu()) == v:
                        acc_v[label[i]] += 1.0
                    if np.asarray(label[i].cpu()) == a:
                        acc_a[label[i]] += 1.0
                # print(num)
        return sum(acc) / sum(num), sum(acc_av) / sum(num), sum(acc_va) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)

    @torch.no_grad()
    def aggregate(self, all_client_params, all_client_data_num, selected_clients):
        weights_cache = list(all_client_data_num)
        weight_sum = sum(weights_cache)
        weights = torch.tensor(weights_cache, device=self.device) / weight_sum

        weights_audio_cache = deepcopy(weights_cache)
        weights_visual_cache = deepcopy(weights_cache)
        for i, sc in enumerate(selected_clients):
            if sc in self.mm_client or sc in self.audio_client:
                weights_audio_cache[i] = weights_cache[i]
            else:
                weights_audio_cache[i] = 0

            if sc in self.mm_client or sc in self.visual_client:
                weights_visual_cache[i] = weights_cache[i]
            else:
                weights_visual_cache[i] = 0

        weight_sum_audio = sum(weights_audio_cache)
        weight_sum_visual = sum(weights_visual_cache)

        if all(w == 0 for w in weights_audio_cache):
            weights_audio = torch.zeros_like(weights).to(self.device)
            audio_update = False
        else:
            weights_audio = torch.tensor(weights_audio_cache, device=self.device) / weight_sum_audio
            audio_update = True
        if all(w == 0 for w in weights_visual_cache):
            weights_visual = torch.zeros_like(weights).to(self.device)
            visual_update = False
        else:
            weights_visual = torch.tensor(weights_visual_cache, device=self.device) / weight_sum_visual
            visual_update = True

        print('weights: ', weights_audio, weights_visual)

        w_audio_avg = deepcopy(all_client_params[0][0])
        w_visual_avg = deepcopy(all_client_params[1][0])
        w_audio_a_avg = deepcopy(all_client_params[2][0])
        w_audio_v_avg = deepcopy(all_client_params[3][0])
        w_visual_a_avg = deepcopy(all_client_params[4][0])
        w_visual_v_avg = deepcopy(all_client_params[5][0])
        w_fusion_avg = deepcopy(all_client_params[6][0])
        w_av_class_avg = deepcopy(all_client_params[7][0])
        w_va_class_avg = deepcopy(all_client_params[8][0])

        for key in w_audio_avg.keys():
            for i in range(len(all_client_params[0])):
                if i == 0:
                    w_audio_avg[key] = weights_audio[i] * all_client_params[0][i][key]
                else:
                    w_audio_avg[key] += weights_audio[i] * all_client_params[0][i][key]
        for key in w_visual_avg.keys():
            for i in range(len(all_client_params[1])):
                if i == 0:
                    w_visual_avg[key] = weights_visual[i] * all_client_params[1][i][key]
                else:
                    w_visual_avg[key] += weights_visual[i] * all_client_params[1][i][key]

        for key in w_audio_a_avg.keys():
            for i in range(len(all_client_params[2])):
                if i == 0:
                    w_audio_a_avg[key] = weights_audio[i] * all_client_params[2][i][key]
                else:
                    w_audio_a_avg[key] += weights_audio[i] * all_client_params[2][i][key]
        for key in w_audio_v_avg.keys():
            for i in range(len(all_client_params[3])):
                if i == 0:
                    w_audio_v_avg[key] = weights_audio[i] * all_client_params[3][i][key]
                else:
                    w_audio_v_avg[key] += weights_audio[i] * all_client_params[3][i][key]

        for key in w_visual_a_avg.keys():
            for i in range(len(all_client_params[4])):
                if i == 0:
                    w_visual_a_avg[key] = weights_visual[i] * all_client_params[4][i][key]
                else:
                    w_visual_a_avg[key] += weights_visual[i] * all_client_params[4][i][key]
        for key in w_visual_v_avg.keys():
            for i in range(len(all_client_params[5])):
                if i == 0:
                    w_visual_v_avg[key] = weights_visual[i] * all_client_params[5][i][key]
                else:
                    w_visual_v_avg[key] += weights_visual[i] * all_client_params[5][i][key]

        for key in w_fusion_avg.keys():
            for i in range(len(all_client_params[2])):
                if i == 0:
                    w_fusion_avg[key] = weights[i] * all_client_params[6][i][key]
                else:
                    w_fusion_avg[key] += weights[i] * all_client_params[6][i][key]

        for key in w_av_class_avg.keys():
            for i in range(len(all_client_params[6])):
                if i == 0:
                    w_av_class_avg[key] = weights_audio[i] * all_client_params[7][i][key]
                else:
                    w_av_class_avg[key] += weights_audio[i] * all_client_params[7][i][key]
        for key in w_va_class_avg.keys():
            for i in range(len(all_client_params[8])):
                if i == 0:
                    w_va_class_avg[key] = weights_visual[i] * all_client_params[8][i][key]
                else:
                    w_va_class_avg[key] += weights_visual[i] * all_client_params[8][i][key]

        return w_audio_avg, w_visual_avg, w_audio_a_avg, w_audio_v_avg, w_visual_a_avg, w_visual_v_avg, w_fusion_avg, \
            w_av_class_avg, w_va_class_avg, audio_update, visual_update

    def run(self):
        self.logger.log("Arguments:", dict(self.args._get_kwargs()))
        self.train()
        self.validate()

