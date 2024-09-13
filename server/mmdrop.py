import os
import pickle
import random
from pprint import pprint

import torch
from path import Path
from rich.console import Console
from rich.progress import track
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from client.base import ClientBase
from client.mmdrop import FedMDropClient
from config.utils import fix_random_seed, allocate_client_modality, process_grad
from data.utils.util import get_val_dataset
from models.basic_model import AVClassifier, DecomposedAVClassifier
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from copy import deepcopy
from .base import ServerBase
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


class FedMDropServer(ServerBase):
    def __init__(self):
        super(FedMDropServer, self).__init__(get_args(), "FedMDrop")

        self.trainer = FedMDropClient(
            args=self.args,
            model=self.model,
            dataset=self.args.dataset,
            batch_size=self.args.batch_size,
            local_epochs=self.args.local_epochs,
            local_lr=self.args.local_lr,
            gpu=self.args.gpu,
        )

        pprint(self.args)

    def train(self):
        print('start training...')
        progress_bar = tqdm(range(self.global_epochs), "Training...")

        f_log = open(self.log_file, 'a')

        for E in progress_bar:
            # select clients participated in training
            # 1. random
            if self.args.clientsel_algo == 'random':
                selected_clients = random.sample(
                    self.client_id_indices, self.args.client_num_per_round
                )
            # 2. submodular
            elif self.args.clientsel_algo == 'submodular':  # select clients with diversity
                if E == 0 or self.args.client_num_per_round == 1:  # at the first epoch or when client_per_round=1, collect gradients from all clients
                    self.all_grads = np.asarray(self.show_grads()[:-1])  # client_num x params_num
                    self.norm_diff = pairwise_distances(self.all_grads, metric="euclidean")  # client_num x client_num
                    np.fill_diagonal(self.norm_diff, 0)
                selected_clients, _ = self.select_cl_submod(E, clients_per_round=self.args.client_num_per_round,
                                                            stochastic_greedy=True)
                # old_grads = self.all_grads.copy()
            elif self.args.clientsel_algo == 'balance_submodular':  # select clients with diversity and also keep modality balance
                if E == 0 or self.args.client_num_per_round == 1:  # at the first epoch or when client_per_round=1, collect gradients from all clients
                    self.all_grads, self.all_audio_grads, self.all_visual_grads, self.all_b_ratios, self.samples = self.show_modalbalance_grads()
                    self.all_grads = np.asarray(self.all_grads[:-1])
                    self.all_audio_grads = np.asarray(self.all_audio_grads[:-1])
                    self.all_visual_grads = np.asarray(self.all_visual_grads[:-1])

                    self.norm_diff = pairwise_distances(self.all_grads, metric="euclidean")  # client_num x client_num
                    np.fill_diagonal(self.norm_diff, 0)
                    self.norm_diff_a = pairwise_distances(self.all_audio_grads,
                                                          metric="euclidean")  # client_num x client_num
                    np.fill_diagonal(self.norm_diff_a, 0)
                    self.norm_diff_v = pairwise_distances(self.all_visual_grads,
                                                          metric="euclidean")  # client_num x client_num
                    np.fill_diagonal(self.norm_diff_v, 0)

                selected_clients, using_modal = self.select_cl_submod(E, clients_per_round=self.args.client_num_per_round,
                                                                      stochastic_greedy=True, balance_method='balansub')
                print('selected clients: ', selected_clients, using_modal)
            elif self.args.clientsel_algo == 'PMR_submodular':  #
                if E == 0 or self.args.client_num_per_round == 1:  # at the first epoch or when client_per_round=1, collect gradients from all clients
                    self.all_grads, self.all_audio_grads, self.all_visual_grads, self.all_b_ratios, self.samples,\
                        _, _ = self.show_PMRbalance_grads()
                    self.global_audio_protos, self.global_visual_protos = self.aggregate_global_proto(self.client_id_indices)
                    self.all_grads = np.asarray(self.all_grads[:-1])
                    self.all_audio_grads = np.asarray(self.all_audio_grads[:-1])
                    self.all_visual_grads = np.asarray(self.all_visual_grads[:-1])

                    self.norm_diff = pairwise_distances(self.all_grads, metric="euclidean")  # client_num x client_num
                    np.fill_diagonal(self.norm_diff, 0)
                    self.norm_diff_a = pairwise_distances(self.all_audio_grads, metric="euclidean")  # client_num x client_num
                    np.fill_diagonal(self.norm_diff_a, 0)
                    self.norm_diff_v = pairwise_distances(self.all_visual_grads, metric="euclidean")  # client_num x client_num
                    np.fill_diagonal(self.norm_diff_v, 0)

                selected_clients, using_modal = self.select_cl_submod(E, clients_per_round=self.args.client_num_per_round,
                                                                      stochastic_greedy=True, balance_method='PMRsub')
                print('selected clients: ', selected_clients, using_modal)
            else:
                pass

            all_audio_params = []
            all_visual_params = []
            all_fusion_params = []
            all_client_data_num = []
            all_audio_drops = []
            all_visual_drops = []

            # global_audio_pts = torch.zeros(self.n_classes, self.args.embed_dim).to(self.device)
            # global_visual_pts = torch.zeros(self.n_classes, self.args.embed_dim).to(self.device)

            audio_c_num = 0
            visual_c_num = 0
            for index, client_id in enumerate(selected_clients):
                print('client {} starts training...'.format(client_id))

                modality_type = self.get_client_type(client_id)
                if self.args.clientsel_algo == 'balance_submodular' or self.args.clientsel_algo == 'PMR_submodular':
                    if modality_type == 'multimodal' and using_modal[index] == 'audio':
                        modality_type = 'audio'
                    elif modality_type == 'multimodal' and using_modal[index] == 'visual':
                        modality_type = 'visual'

                if modality_type == 'audio' or modality_type == 'multimodal':
                    audio_c_num += 1
                if modality_type == 'visual' or modality_type == 'multimodal':
                    visual_c_num += 1

                if self.args.clientsel_algo == 'PMR_submodular':
                    all_params, data_num_client, [a_drop, v_drop], client_grads,\
                        ratio = self.trainer.train(
                        client_id=client_id,
                        model=self.model,
                        modality_type=modality_type,
                        global_epoch=E,
                        global_proto=[self.global_audio_protos, self.global_visual_protos]
                    )

                    # global_audio_pts = torch.add(global_audio_pts, local_audio_proto * data_num_client)
                    # global_visual_pts = torch.add(global_visual_pts, local_visual_proto * data_num_client)
                else:
                    all_params, data_num_client, [a_drop, v_drop], client_grads, ratio = self.trainer.train(
                        client_id=client_id,
                        model=self.model,
                        modality_type=modality_type,
                        global_epoch=E,
                        global_proto=None
                    )

                if self.args.clientsel_algo == 'balance_submodular' or self.args.clientsel_algo == 'PMR_submodular':
                    if using_modal[index] == 'audio':
                        a_drop = False
                        v_drop = True
                    elif using_modal[index] == 'visual':
                        a_drop = True
                        v_drop = False

                if self.args.clientsel_algo == 'submodular':
                    self.all_grads[client_id] = client_grads
                elif self.args.clientsel_algo == 'balance_submodular' or self.args.clientsel_algo == 'PMR_submodular':
                    if using_modal[index] == 'multi':
                        self.all_grads[client_id] = client_grads
                    elif using_modal[index] == 'visual':
                        self.all_visual_grads[client_id] = client_grads
                    elif using_modal[index] == 'audio':
                        self.all_audio_grads[client_id] = client_grads
                    self.all_b_ratios[client_id] = ratio

                all_audio_params.append(all_params[0])
                all_visual_params.append(all_params[1])
                all_fusion_params.append(all_params[2])
                all_client_data_num.append(data_num_client)
                all_audio_drops.append(a_drop)
                all_visual_drops.append(v_drop)

            # update gradient metrix
            if self.args.clientsel_algo == 'submodular':
                self.norm_diff[selected_clients] = pairwise_distances(self.all_grads[selected_clients], self.all_grads,
                                                                      metric="euclidean")
                self.norm_diff[:, selected_clients] = self.norm_diff[selected_clients].T
            elif self.args.clientsel_algo == 'balance_submodular':
                self.norm_diff[selected_clients] = pairwise_distances(self.all_grads[selected_clients], self.all_grads,
                                                                      metric="euclidean")
                self.norm_diff[:, selected_clients] = self.norm_diff[selected_clients].T
                self.norm_diff_a[selected_clients] = pairwise_distances(self.all_audio_grads[selected_clients], self.all_audio_grads,
                                                                      metric="euclidean")
                self.norm_diff_a[:, selected_clients] = self.norm_diff_a[selected_clients].T
                self.norm_diff_v[selected_clients] = pairwise_distances(self.all_visual_grads[selected_clients], self.all_visual_grads,
                                                                      metric="euclidean")
                self.norm_diff_v[:, selected_clients] = self.norm_diff_v[selected_clients].T
                self.all_b_ratios[-1] = np.dot(np.asarray(self.all_b_ratios[:-1]), np.asarray(self.samples)) / np.sum(np.asarray(self.samples))
            elif self.args.clientsel_algo == 'PMR_submodular':
                self.norm_diff[selected_clients] = pairwise_distances(self.all_grads[selected_clients], self.all_grads,
                                                                      metric="euclidean")
                self.norm_diff[:, selected_clients] = self.norm_diff[selected_clients].T
                self.norm_diff_a[selected_clients] = pairwise_distances(self.all_audio_grads[selected_clients], self.all_audio_grads,
                                                                      metric="euclidean")
                self.norm_diff_a[:, selected_clients] = self.norm_diff_a[selected_clients].T
                self.norm_diff_v[selected_clients] = pairwise_distances(self.all_visual_grads[selected_clients], self.all_visual_grads,
                                                                      metric="euclidean")
                self.norm_diff_v[:, selected_clients] = self.norm_diff_v[selected_clients].T
                self.all_b_ratios[-1] = np.dot(np.asarray(self.all_b_ratios[:-1]), np.asarray(self.samples)) / np.sum(np.asarray(self.samples))

                # self.global_audio_protos = global_audio_pts * 1.0 / sum(all_client_data_num)
                # self.global_visual_protos = global_visual_pts * 1.0 / sum(all_client_data_num)

            # aggregation
            global_audio_weights, global_visual_weights, global_fusion_weights, audio_update, visual_update = \
                self.aggregate([all_audio_params, all_visual_params, all_fusion_params], all_client_data_num,
                               all_audio_drops, all_visual_drops, selected_clients)
            if audio_update:
                if self.args.dataset == 'CREMAD' or self.args.dataset == 'AVE':
                    self.model.audio_net.load_state_dict(global_audio_weights)
                elif self.args.dataset == 'CGMNIST':
                    self.model.gray_net.load_state_dict(global_audio_weights)
                elif self.args.dataset == 'CrisisMMD':
                    self.model.imageEncoder.load_state_dict(global_audio_weights)
                elif self.args.dataset == 'ModelNet40':
                    self.model.flow_net.load_state_dict(global_audio_weights)
            if visual_update:
                if self.args.dataset == 'CREMAD' or self.args.dataset == 'AVE' or self.args.dataset == 'ModelNet40':
                    self.model.visual_net.load_state_dict(global_visual_weights)
                elif self.args.dataset == 'CGMNIST':
                    self.model.colored_net.load_state_dict(global_visual_weights)
                elif self.args.dataset == 'CrisisMMD':
                    self.model.textEncoder.load_state_dict(global_visual_weights)
            if self.args.dataset == 'CrisisMMD':
                self.model.cls_layer.load_state_dict(global_fusion_weights)
            else:
                self.model.fusion_module.load_state_dict(global_fusion_weights)

            if self.args.clientsel_algo == 'PMR_submodular':
                self.global_audio_protos, self.global_visual_protos = self.aggregate_global_proto(selected_clients)

            acc, acc_a, acc_v = self.validate()
            print('accuracy for round {}: '.format(E), acc, acc_a, acc_v)

            if self.args.clientsel_algo == 'balance_submodular' or self.args.clientsel_algo == 'PMR_submodular':
                f_log.write(str(E) +
                            "\t" + str(acc) +
                            "\t" + str(acc_a) +
                            "\t" + str(acc_v) +
                            "\t" + str(self.all_b_ratios[-1]) +
                            "\t" + str(acc_a/acc_v) +
                            "\t" + str(audio_c_num) +
                            "\t" + str(visual_c_num) +
                            "\n")
            else:
                f_log.write(str(E) +
                            "\t" + str(acc) +
                            "\t" + str(acc_a) +
                            "\t" + str(acc_v) +
                            "\n")
            f_log.flush()

            if E % self.args.save_period == 0:
                if self.args.dataset == 'CREMAD' or self.args.dataset == 'AVE':
                    torch.save(
                        self.model.audio_net.state_dict(),
                        self.save_dir + "/global_model_audio.pt",
                    )
                    torch.save(
                        self.model.visual_net.state_dict(),
                        self.save_dir + "/global_model_visual.pt",
                    )
                elif self.args.dataset == 'CGMNIST':
                    torch.save(
                        self.model.gray_net.state_dict(),
                        self.save_dir + "/global_model_gray.pt",
                    )
                    torch.save(
                        self.model.colored_net.state_dict(),
                        self.save_dir + "/global_model_colored.pt",
                    )
                elif self.args.dataset == 'CrisisMMD':
                    torch.save(
                        self.model.imageEncoder.state_dict(),
                        self.save_dir + "/global_model_audio.pt",
                    )
                    torch.save(
                        self.model.textEncoder.state_dict(),
                        self.save_dir + "/global_model_visual.pt",
                    )
                elif self.args.dataset == 'ModelNet40':
                    torch.save(
                        self.model.flow_net.state_dict(),
                        self.save_dir + "/global_model_audio.pt",
                    )
                    torch.save(
                        self.model.visual_net.state_dict(),
                        self.save_dir + "/global_model_visual.pt",
                    )
                if self.args.dataset == 'CrisisMMD':
                    torch.save(
                        self.model.cls_layer.state_dict(),
                        self.save_dir + "/global_model_fusion.pt",
                    )
                else:
                    torch.save(
                        self.model.fusion_module.state_dict(),
                        self.save_dir + "/global_model_fusion.pt",
                    )
                with open(self.save_dir + "/epoch.pkl", "wb") as f:
                    pickle.dump(E, f)
        f_log.close()

    def aggregate_global_proto(self, selected_clients):
        # cal global prototypes
        if self.args.dataset == 'CrisisMMD':
            audio_prototypes = torch.zeros(self.n_classes, self.args.dim_img_repr).to(self.device)
            visual_prototypes = torch.zeros(self.n_classes, self.args.dim_text_repr).to(self.device)
        else:
            audio_prototypes = torch.zeros(self.n_classes, self.args.embed_dim).to(self.device)
            visual_prototypes = torch.zeros(self.n_classes, self.args.embed_dim).to(self.device)
        a_protos = []
        v_protos = []
        data_nums_a = []
        data_nums_v = []
        modality_types = []
        for client in selected_clients:
            if client in self.mm_client:
                modality_type = 'multimodal'
            elif client in self.audio_client:
                modality_type = 'audio'
            elif client in self.visual_client:
                modality_type = 'visual'
            else:
                raise ValueError('Non-exist modality type')
            a_proto, v_proto, count_class = self.trainer.get_local_prototype(client, self.model)
            # if modality_type == 'audio':
            #     v_proto = torch.zeros(self.n_classes, self.args.embed_dim).to(self.device)
            # elif modality_type == 'visual':
            #     a_proto = torch.zeros(self.n_classes, self.args.embed_dim).to(self.device)
            a_protos.append(a_proto)
            v_protos.append(v_proto)
            if modality_type == 'multimodal':
                count_class_a = deepcopy(count_class)
                count_class_v = deepcopy(count_class)
            elif modality_type == 'audio':
                count_class_a = deepcopy(count_class)
                count_class_v = [0 for _ in range(self.n_classes)]
            elif modality_type == 'visual':
                count_class_a = [0 for _ in range(self.n_classes)]
                count_class_v = deepcopy(count_class)
            data_nums_a.append(count_class_a)
            data_nums_v.append(count_class_v)
            modality_types.append(modality_type)
        data_nums_a = np.array(data_nums_a)
        data_nums_v = np.array(data_nums_v)

        for idx, mt in enumerate(modality_types):
            for c in range(self.n_classes):
                if sum(data_nums_a[:, c]) == 0:
                    audio_prototypes[c] += 0 * a_protos[idx][c]
                else:
                    audio_prototypes[c] += data_nums_a[idx, c] / sum(data_nums_a[:, c]) * a_protos[idx][c]
                if sum(data_nums_v[:, c]) == 0:
                    visual_prototypes[c] += 0 * v_protos[idx][c]
                else:
                    visual_prototypes[c] += data_nums_v[idx, c] / sum(data_nums_v[:, c]) * v_protos[idx][c]
        return audio_prototypes, visual_prototypes


    @torch.no_grad()
    def aggregate(self, all_client_params, all_client_data_num, all_audio_drops, all_visual_drops, selected_clients):
        weights_cache = list(all_client_data_num)
        weight_sum = sum(weights_cache)
        weights = torch.tensor(weights_cache, device=self.device) / weight_sum

        weights_audio_cache = deepcopy(weights_cache)
        weights_visual_cache = deepcopy(weights_cache)
        for i, sc in enumerate(selected_clients):
            if (sc in self.mm_client or sc in self.audio_client) and all_audio_drops[i] is False:
                weights_audio_cache[i] = weights_cache[i]
            else:
                weights_audio_cache[i] = 0

            if (sc in self.mm_client or sc in self.visual_client) and all_visual_drops[i] is False:
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

    def run(self):
        self.train()
        self.validate()
        # if self.args.log:
        #     if not os.path.isdir(LOG_DIR):
        #         os.mkdir(LOG_DIR)
        #     self.logger.save_html(LOG_DIR / self.log_name)

        # delete all temporary files
        # if os.listdir(self.save_dir) != []:
        #     os.system(f"rm -rf {self.save_dir}")
