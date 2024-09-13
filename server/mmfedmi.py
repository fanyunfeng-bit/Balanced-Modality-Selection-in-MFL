# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 2023
fedavg with modality imbalance technique (server)

@author: fyf
"""

import os
import pickle
import random
from collections import OrderedDict

import torch
from path import Path
from rich.console import Console
from rich.progress import track
from tqdm import tqdm

# sys.path.append('..')
from client.base import ClientBase
from client.mmfedmi import FedMIClient
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


class FedMIServer(ServerBase):
    def __init__(self):
        super(FedMIServer, self).__init__(get_args(), "FedMI")
        fix_random_seed(self.args.seed)

        self.trainer = FedMIClient(args=self.args,
                                   model=deepcopy(self.model),
                                   dataset=self.args.dataset,
                                   batch_size=self.args.batch_size,
                                   local_epochs=self.args.local_epochs,
                                   local_lr=self.args.local_lr,
                                   logger=self.logger,
                                   gpu=self.args.gpu, )

        self.global_audio_proto = []
        self.global_visual_proto = []

    def train(self):
        print('start training...')
        progress_bar = tqdm(range(self.global_epochs), "Training...")

        f_log = open(self.log_file, 'a')

        for E in progress_bar:
            selected_clients = random.sample(
                self.client_id_indices, self.args.client_num_per_round
            )
            # selected_clients = [0]

            all_audio_params = []
            all_visual_params = []
            all_fusion_params = []
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
                    global_proto = [self.global_audio_proto, self.global_visual_proto]
                all_params, data_num_client = self.trainer.train(
                    client_id=client_id,
                    model=deepcopy(self.model),
                    modality_type=modality_type,
                    global_epoch=E,
                    global_proto=global_proto
                )

                all_audio_params.append(deepcopy(all_params[0]))
                all_visual_params.append(deepcopy(all_params[1]))
                all_fusion_params.append(deepcopy(all_params[2]))
                all_client_data_num.append(data_num_client)

            # cal global prototypes
            audio_prototypes = torch.zeros(self.n_classes, self.args.embed_dim).to(self.device)
            visual_prototypes = torch.zeros(self.n_classes, self.args.embed_dim).to(self.device)
            data_nums = 0
            for client in selected_clients:
                a_proto, v_proto, data_num = self.trainer.get_local_prototype(client, self.model)
                audio_prototypes += a_proto * data_num
                visual_prototypes += v_proto * data_num
                data_nums += data_num
            audio_prototypes /= data_nums
            visual_prototypes /= data_nums
            # 可以是momentum更新
            self.global_audio_proto = audio_prototypes
            self.global_visual_proto = visual_prototypes

            # aggregation
            global_audio_weights, global_visual_weights, global_fusion_weights, audio_update, visual_update = \
                self.aggregate([all_audio_params, all_visual_params, all_fusion_params], all_client_data_num,
                               selected_clients)
            if audio_update:
                self.model.audio_net.load_state_dict(global_audio_weights)
            if visual_update:
                self.model.visual_net.load_state_dict(global_visual_weights)
            self.model.fusion_module.load_state_dict(global_fusion_weights)

            acc = self.validate()
            print('accuracy for round {}: '.format(E), acc)

            f_log.write(str(E) +
                        "\t" + str(acc) +
                        "\n")
            f_log.flush()

            if E % self.args.save_period == 0:
                torch.save(
                    self.model.audio_net.state_dict(),
                    self.save_dir / "global_model_audio.pt",
                )
                torch.save(
                    self.model.visual_net.state_dict(),
                    self.save_dir / "global_model_visual.pt",
                )
                torch.save(
                    self.model.fusion_module.state_dict(),
                    self.save_dir / "global_model_fusion.pt",
                )
                with open(self.save_dir / "epoch.pkl", "wb") as f:
                    pickle.dump(E, f)
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

        with torch.no_grad():
            for step, (spec, image, label) in enumerate(val_dataloader):
                spec = spec.to(self.device)
                image = image.to(self.device)
                label = label.to(self.device)

                if self.args.visual_only:
                    out = self.model.forward_visual(image.float(), bsz=label.shape[0])
                elif self.args.audio_only:
                    out = self.model.forward_audio(spec.unsqueeze(1).float())
                else:
                    if self.args.dataset != 'CGMNIST':
                        a, v, out = self.model(spec.unsqueeze(1).float(), image.float())
                    else:
                        a, v, out = self.model(spec, image)  # gray colored

                # if args.fusion_method == 'sum':
                #     out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_y.weight, 0, 1)) +
                #              model.fusion_module.fc_y.bias)
                #     out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_x.weight, 0, 1)) +
                #              model.fusion_module.fc_x.bias)
                # elif args.fusion_method == 'concat':
                #     weight_size = model.fusion_module.fc_out.weight.size(1)
                #     out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
                #              + model.fusion_module.fc_out.bias / 2)
                #     out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
                #              + model.fusion_module.fc_out.bias / 2)
                # elif args.fusion_method == 'film':
                #     out_v = out
                #     out_a = out

                prediction = softmax(out)
                # pred_v = softmax(out_v)
                # pred_a = softmax(out_a)
                # print('prediction: ', prediction)
                for i in range(image.shape[0]):

                    ma = np.argmax(prediction[i].cpu().data.numpy())
                    # v = np.argmax(pred_v[i].cpu().data.numpy())
                    # a = np.argmax(pred_a[i].cpu().data.numpy())
                    num[label[i]] += 1.0

                    # pdb.set_trace()
                    if np.asarray(label[i].cpu()) == ma:
                        acc[label[i]] += 1.0
                    # if np.asarray(label[i].cpu()) == v:
                    #     acc_v[label[i]] += 1.0
                    # if np.asarray(label[i].cpu()) == a:
                    #     acc_a[label[i]] += 1.0
                # print(num)
        return sum(acc) / sum(num)

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
        w_fusion_avg = deepcopy(all_client_params[2][0])

        for key in w_audio_avg.keys():
            for i in range(len(all_client_params[0])):
                if i == 0:
                    w_audio_avg[key] = weights_audio[i] * all_client_params[0][i][key]
                else:
                    w_audio_avg[key] += weights_audio[i] * all_client_params[0][i][key]
            # w_audio_avg[key] = torch.div(w_audio_avg[key], len(all_client_params[0]))
        for key in w_visual_avg.keys():
            for i in range(len(all_client_params[1])):
                if i == 0:
                    w_visual_avg[key] = weights_visual[i] * all_client_params[1][i][key]
                else:
                    w_visual_avg[key] += weights_visual[i] * all_client_params[1][i][key]

            # w_visual_avg[key] = torch.div(w_visual_avg[key], len(all_client_params[1]))

        for key in w_fusion_avg.keys():
            for i in range(len(all_client_params[2])):
                if i == 0:
                    w_fusion_avg[key] = weights[i] * all_client_params[2][i][key]
                else:
                    w_fusion_avg[key] += weights[i] * all_client_params[2][i][key]
            # w_fusion_avg[key] = torch.div(w_fusion_avg[key], len(all_client_params[2]))

        return w_audio_avg, w_visual_avg, w_fusion_avg, audio_update, visual_update

    def test(self) -> None:
        all_loss = []
        all_correct = []
        all_samples = []
        for client_id in track(
                self.client_id_indices,
                "[bold blue]Testing...",
                console=self.logger,
                disable=self.args.log,
        ):
            client_local_params = clone_parameters(self.global_params_dict)
            stats = self.trainer.test(
                client_id=client_id,
                model_params=client_local_params,
            )

            all_loss.append(stats["loss"])
            all_correct.append(stats["correct"])
            all_samples.append(stats["size"])
        # self.logger.log("=" * 20, "RESULTS", "=" * 20, style="bold green")
        # self.logger.log(
        #     "loss: {:.4f}    accuracy: {:.2f}%".format(
        #         sum(all_loss) / sum(all_samples),
        #         sum(all_correct) / sum(all_samples) * 100.0,
        #     )
        # )

        acc_range = [90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0]
        min_acc_idx = 10
        max_acc = 0
        for E, (corr, n) in enumerate(zip(self.num_correct, self.num_samples)):
            avg_acc = sum(corr) / sum(n) * 100.0
            for i, acc in enumerate(acc_range):
                if avg_acc >= acc and avg_acc > max_acc:
                    self.logger.log(
                        "{} achieved {}% accuracy({:.2f}%) at epoch: {}".format(
                            self.algo, acc, avg_acc, E
                        )
                    )
                    max_acc = avg_acc
                    min_acc_idx = i
                    break
            acc_range = acc_range[:min_acc_idx]

    def run(self):
        self.logger.log("Arguments:", dict(self.args._get_kwargs()))
        self.train()
        self.validate()
        # if self.args.log:
        #     if not os.path.isdir(LOG_DIR):
        #         os.mkdir(LOG_DIR)
        #     self.logger.save_html(LOG_DIR / self.log_name)

        # delete all temporary files
        # if os.listdir(self.save_dir) != []:
        #     os.system(f"rm -rf {self.save_dir}")
