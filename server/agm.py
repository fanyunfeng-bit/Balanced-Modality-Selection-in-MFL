import os
import pickle
import random
from collections import OrderedDict

import torch
from path import Path
from rich.console import Console
from rich.progress import track
from tqdm import tqdm

from client.agm import AGMClient
# sys.path.append('..')
from client.base import ClientBase
from config.utils import fix_random_seed, allocate_client_modality, EU_dist, dot_product_angle_tensor, \
    grad_amplitude_diff, calculate_prototype, process_grad
from data.utils.util import get_val_dataset
from models.agm_model import GradMod
from models.basic_model import AVClassifier, DecomposedAVClassifier, CGClassifier, DenseNetBertMMModel
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from copy import deepcopy
import math


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class AGMServer:
    def __init__(self, args, algo: str):
        self.args = args
        self.algo = algo

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

        self.device = torch.device("cuda" if self.args.gpu and torch.cuda.is_available() else "cpu")

        fix_random_seed(self.args.seed)

        self.model = GradMod(self.args)
        self.model.apply(weight_init)
        self.model.to(self.device)

        self.client_id_indices = [i for i in range(self.args.client_num)]
        self.client_num_in_total = self.args.client_num

        # allocate each client with different modalities
        self.multi_ratio = self.args.multi_ratio

        self.mm_client, self.audio_client, self.visual_client = \
            allocate_client_modality(self.client_id_indices, self.multi_ratio, self.args.audio_only, self.args.visual_only)

        self.this_file_dir = Path(__file__).parent.parent.abspath()
        self.save_dir = self.this_file_dir / 'checkpoints' / self.algo / self.args.dataset
        self.log_dir = self.this_file_dir / 'logs' / self.algo / self.args.dataset
        if self.algo != 'FedMDrop':
            self.save_dir = os.path.join(self.save_dir, 'CN{}_{}-LE{}-LR{}-MR{}-alpha{}-MDrop_{}-CS_{}-Bthresh{}-Proto_{}-OGM_{}-PMR_{}'.format(self.args.client_num, self.args.client_num_per_round,
                                            self.args.local_epochs, self.args.local_lr, self.multi_ratio, self.args.alpha, str(self.args.modality_drop), self.args.clientsel_algo, self.args.balansubmod_thresh,
                                            str(self.args.fedproto), str(self.args.OGM), str(self.args.PMR)))
        else:
            self.save_dir = os.path.join(self.save_dir,
                                         'CN{}_{}-LE{}-LR{}-MR{}-alpha{}-MDrop_{}-CS_{}-Bthresh{}'.format(
                                             self.args.client_num, self.args.client_num_per_round,
                                             self.args.local_epochs, self.args.local_lr, self.multi_ratio,
                                             self.args.alpha, str(self.args.modality_drop), self.args.clientsel_algo,
                                             self.args.balansubmod_thresh,))
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if self.algo != 'FedMDrop':
            self.log_file = self.log_dir + '/log-CN{}_{}-LE{}-LR{}-MR{}-alpha{}-MDrop_{}-CS_{}-Bthresh{}-Proto_{}-OGM_{}-PMR_{}.txt'.format(self.args.client_num, self.args.client_num_per_round,
                                            self.args.local_epochs, self.args.local_lr, self.multi_ratio, self.args.alpha, str(self.args.modality_drop), self.args.clientsel_algo, self.args.balansubmod_thresh,
                                            str(self.args.fedproto), str(self.args.OGM), str(self.args.PMR))
        else:
            self.log_file = self.log_dir + '/log-CN{}_{}-LE{}-LR{}-MR{}-alpha{}-MDrop_{}-CS_{}-Bthresh{}-ratio.txt'.format(
                self.args.client_num, self.args.client_num_per_round,
                self.args.local_epochs, self.args.local_lr, self.multi_ratio, self.args.alpha,
                str(self.args.modality_drop), self.args.clientsel_algo, self.args.balansubmod_thresh,)

        passed_epoch = 0

        if os.listdir(self.save_dir) != [] and self.args.save_period > 0 and self.args.load:
            if self.algo != 'FedCMD' and self.algo != 'FedCMI':
                if os.path.exists(self.save_dir):
                    global_audio_params_dict = torch.load(self.save_dir / "global_model_audio.pt")
                    global_visual_params_dict = torch.load(self.save_dir / "global_model_visual.pt")
                    global_fusion_params_dict = torch.load(self.save_dir / "global_model_fusion.pt")
                    self.model.audio_net.load_state_dict(global_audio_params_dict)
                    self.model.visual_net.load_state_dict(global_visual_params_dict)
                    self.model.fusion_module.load_state_dict(global_fusion_params_dict)
                    print("Find existed global model...")

            if os.path.exists(self.save_dir / "epoch.pkl"):
                with open(self.save_dir / "epoch.pkl", "rb") as f:
                    passed_epoch = pickle.load(f)
                print(f"Have run {passed_epoch} epochs already.", )
        else:
            if (os.path.isfile(self.log_file)):
                os.remove(self.log_file)  # 删掉已有同名文件

        self.global_epochs = self.args.global_epochs - passed_epoch

        self.trainer = AGMClient(
            args=self.args,
            model=self.model,
            dataset=self.args.dataset,
            batch_size=self.args.batch_size,
            local_epochs=self.args.local_epochs,
            local_lr=self.args.local_lr,
            gpu=self.args.gpu,
        )

        self.num_correct = [[] for _ in range(self.global_epochs)]
        self.num_samples = [[] for _ in range(self.global_epochs)]

        self.norm_diff = np.zeros((len(self.client_id_indices), len(self.client_id_indices)))
        self.norm_diff_a = np.zeros((len(self.client_id_indices), len(self.client_id_indices)))
        self.norm_diff_v = np.zeros((len(self.client_id_indices), len(self.client_id_indices)))

        self.all_b_ratios = np.zeros(len(self.client_id_indices)+1)  # 记录每个client和总模型的不均衡程度
        self.samples = np.zeros(len(self.client_id_indices))  # 每个client的样本数量
        if self.args.dataset != 'CrisisMMD':
            self.global_audio_protos = torch.zeros(self.n_classes, args.embed_dim).to(self.device)
            self.global_visual_protos = torch.zeros(self.n_classes, args.embed_dim).to(self.device)
        else:
            self.global_audio_protos = torch.zeros(self.n_classes, args.dim_img_repr).to(self.device)  # image
            self.global_visual_protos = torch.zeros(self.n_classes, args.dim_text_repr).to(self.device)  # text

    def select_powd(self):
        init_set = random.sample(
            self.client_id_indices, self.args.init_set_client_num
        )
        loss_list = []
        for client_id in init_set:
            if client_id in self.mm_client:
                modality_type = 'multimodal'
            elif client_id in self.audio_client:
                modality_type = 'audio'  # gray
            elif client_id in self.visual_client:
                modality_type = 'visual'  # color
            else:
                raise ValueError('Non-exist modality type')


            local_loss = self.trainer.get_local_loss(
                client_id=client_id,
                model=self.model,
                modality_type=modality_type,
            )
            loss_list.append(local_loss)
        loss_list = np.asarray(loss_list)
        selected_clients = init_set[loss_list.argsort()[-self.args.client_num_per_round:]]
        return selected_clients


    def train(self):
        print('start training...')
        progress_bar = tqdm(range(self.global_epochs), "Training...")

        f_log = open(self.log_file, 'a')

        for E in progress_bar:

            # select clients for each round
            if self.algo == 'PowD':
                selected_clients = self.select_powd()
            else:
                if self.args.clientsel_algo == 'random':
                    selected_clients = random.sample(
                        self.client_id_indices, self.args.client_num_per_round
                    )
                    # selected_clients = [0]
                # 2. submodular
                else:  # select clients with diversity
                    pass

            all_audio_params = []
            all_visual_params = []
            all_fusion_params = []
            all_client_data_num = []
            all_model_params = []

            each_epoch_score_a = [0. for _ in range(self.args.client_num)]
            each_epoch_score_v = [0. for _ in range(self.args.client_num)]
            audio_lr_ratio = [1.0 for _ in range(self.args.client_num)]
            visual_lr_ratio = [1.0 for _ in range(self.args.client_num)]

            # local training for each client
            for client_id in selected_clients:
                print('client {} starts training...'.format(client_id))
                if client_id in self.mm_client:
                    modality_type = 'multimodal'
                elif client_id in self.audio_client:
                    modality_type = 'audio'  # gray
                elif client_id in self.visual_client:
                    modality_type = 'visual'  # color
                else:
                    raise ValueError('Non-exist modality type')

                each_epoch_score_a[client_id], each_epoch_score_v[client_id], all_params, data_num_client = self.trainer.train(
                    client_id=client_id,
                    model=deepcopy(self.model),
                    modality_type=modality_type,
                    global_epoch=E,
                    last_score_a=each_epoch_score_a[client_id],
                    last_score_v=each_epoch_score_v[client_id],
                    audio_lr_ratio=audio_lr_ratio[client_id],
                    visual_lr_ratio=visual_lr_ratio[client_id]
                )


                all_model_params.append(deepcopy(all_params))
                # all_visual_params.append(deepcopy(all_params[1]))
                # all_fusion_params.append(deepcopy(all_params[2]))
                all_client_data_num.append(data_num_client)

            # aggregation
            global_weights, audio_update, visual_update = \
                self.aggregate(all_model_params, all_client_data_num, selected_clients)

            self.model.load_state_dict(global_weights)

            acc, acc_a, acc_v, audio_loss, visual_loss = self.validate()
            print('accuracy for round {}: '.format(E), acc, acc_a, acc_v)

            f_log.write(str(E) +
                        "\t" + str(acc) +
                        "\t" + str(acc_a) +
                        "\t" + str(acc_v) +
                        "\n")
            f_log.flush()

            # if E % self.args.save_period == 0:
            #     if self.args.dataset == 'CREMAD' or self.args.dataset == 'AVE':
            #         torch.save(
            #             self.model.audio_net.state_dict(),
            #             self.save_dir + "/global_model_audio.pt",
            #         )
            #         torch.save(
            #             self.model.visual_net.state_dict(),
            #             self.save_dir + "/global_model_visual.pt",
            #         )
            #     elif self.args.dataset == 'CGMNIST':
            #         torch.save(
            #             self.model.gray_net.state_dict(),
            #             self.save_dir + "/global_model_gray.pt",
            #         )
            #         torch.save(
            #             self.model.colored_net.state_dict(),
            #             self.save_dir + "/global_model_colored.pt",
            #         )
            #     elif self.args.dataset == 'CrisisMMD':
            #         torch.save(
            #             self.model.imageEncoder.state_dict(),
            #             self.save_dir + "/global_model_audio.pt",
            #         )
            #         torch.save(
            #             self.model.textEncoder.state_dict(),
            #             self.save_dir + "/global_model_visual.pt",
            #         )
            #     if self.args.dataset == 'CrisisMMD':
            #         torch.save(
            #             self.model.cls_layer.state_dict(),
            #             self.save_dir + "/global_model_fusion.pt",
            #         )
            #     else:
            #         torch.save(
            #             self.model.fusion_module.state_dict(),
            #             self.save_dir + "/global_model_fusion.pt",
            #         )
            #     with open(self.save_dir + "/epoch.pkl", "wb") as f:
            #         pickle.dump(E, f)
        f_log.close()

    def validate(self):
        val_dataset = get_val_dataset(self.args, self.args.dataset)
        test_dataloader = DataLoader(val_dataset, batch_size=self.args.val_batch_size, shuffle=False, pin_memory=False)

        softmax = nn.Softmax(dim=1)
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            self.model.eval()

            self.model.mode = 'eval'
            num = [0.0 for _ in range(self.n_classes)]
            acc = [0.0 for _ in range(self.n_classes)]
            acc_a = [0.0 for _ in range(self.n_classes)]
            acc_v = [0.0 for _ in range(self.n_classes)]
            valid_score_a = 0.
            valid_score_v = 0.
            test_loss = 0.
            test_audio_loss = 0.
            test_visual_loss = 0.
            total_batch = len(test_dataloader)

            for step, (spec, image, label) in enumerate(test_dataloader):
                spec = spec.to(self.device)
                image = image.to(self.device)
                label = label.to(self.device)

                if self.args.fl_method == "AGM":
                    total_out, pad_visual_out, pad_audio_out, zero_padding_out, out = self.model(
                        spec.unsqueeze(1).float(), image.float())
                    out_a = 0.5 * (total_out - pad_audio_out + pad_visual_out)
                    out_v = 0.5 * (total_out - pad_visual_out + pad_audio_out)
                else:
                    out_a, out_v, out = self.model(spec.unsqueeze(1).float(), image.float())

                prediction = softmax(out)
                pred_a = softmax(out_a)
                pred_v = softmax(out_v)

                for j in range(image.shape[0]):
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

        accuracy = sum(acc) / sum(num)
        accuracy_a = sum(acc_a) / sum(num)
        accuracy_v = sum(acc_v) / sum(num)

        self.model.mode = 'train'
        return accuracy, accuracy_a, accuracy_v, test_audio_loss, test_visual_loss

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

        w_model_avg = deepcopy(all_client_params[0])

        for key in w_model_avg.keys():
            for i in range(len(all_client_params)):
                if i == 0:
                    w_model_avg[key] = weights[i] * all_client_params[i][key]
                else:
                    w_model_avg[key] += weights[i] * all_client_params[i][key]

        return w_model_avg, audio_update, visual_update

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

    def show_grads(self):
        """
        :return:
        gradients on all clients and the global gradients
        """
        model_len = process_grad(self.model.parameters()).shape[0]
        global_grads = np.zeros(model_len)

        cc = 0
        samples = []

        for c_id in self.client_id_indices:
            modality_type = self.get_client_type(c_id)
            num_samples, client_grads = self.trainer.get_grads(
                client_id=c_id,
                model=self.model,
                modality_type=modality_type,
                model_len=model_len
            )
            samples.append(num_samples)
            if cc == 0:
                intermediate_grads = np.zeros([len(self.client_id_indices) + 1, len(client_grads)])
            global_grads = np.add(global_grads, client_grads * num_samples)
            intermediate_grads[cc] = client_grads
            cc += 1
        global_grads = global_grads * 1.0 / np.sum(np.asarray(samples))
        intermediate_grads[-1] = global_grads

        return intermediate_grads

    def show_modalbalance_grads(self):
        model_len = process_grad(self.model.parameters()).shape[0]
        if self.args.dataset != 'CGMNIST':
            model_len_a = process_grad(self.model.audio_net.parameters()).shape[0]
            model_len_v = process_grad(self.model.visual_net.parameters()).shape[0]
        else:
            model_len_a = process_grad(self.model.gray_net.parameters()).shape[0]
            model_len_v = process_grad(self.model.colored_net.parameters()).shape[0]
        global_grads = np.zeros(model_len)
        audio_grads = np.zeros(model_len_a)
        visual_grads = np.zeros(model_len_v)
        balance_ratios = 0

        cc = 0
        samples = []

        for c_id in self.client_id_indices:
            print('cal the grads of client {}'.format(c_id))
            modality_type = self.get_client_type(c_id)
            num_samples, client_grads, ca_grads, cv_grads, b_ratio = self.trainer.get_modal_grads(
                client_id=c_id,
                model=self.model,
                modality_type=modality_type,
                model_len=model_len
            )
            samples.append(num_samples)
            if cc == 0:
                intermediate_grads = np.zeros([len(self.client_id_indices) + 1, len(client_grads)])
                intermediate_grads_a = np.zeros([len(self.client_id_indices) + 1, len(ca_grads)])
                intermediate_grads_v = np.zeros([len(self.client_id_indices) + 1, len(cv_grads)])
                intermediate_ratios = []
            global_grads = np.add(global_grads, client_grads * num_samples)
            audio_grads = np.add(audio_grads, ca_grads * num_samples)
            visual_grads = np.add(visual_grads, cv_grads * num_samples)
            balance_ratios += b_ratio * num_samples
            intermediate_grads[cc] = client_grads
            intermediate_grads_a[cc] = ca_grads
            intermediate_grads_v[cc] = cv_grads
            intermediate_ratios.append(b_ratio)
            cc += 1
        global_grads = global_grads * 1.0 / np.sum(np.asarray(samples))
        intermediate_grads[-1] = global_grads
        audio_grads = audio_grads * 1.0 / np.sum(np.asarray(samples))
        intermediate_grads_a[-1] = audio_grads
        visual_grads = visual_grads * 1.0 / np.sum(np.asarray(samples))
        intermediate_grads_v[-1] = visual_grads
        balance_ratios = balance_ratios * 1.0 / np.sum(np.asarray(samples))
        intermediate_ratios.append(balance_ratios)

        return intermediate_grads, intermediate_grads_a, intermediate_grads_v, np.asarray(intermediate_ratios), np.asarray(samples)

    def show_PMRbalance_grads(self):
        model_len = process_grad(self.model.parameters()).shape[0]
        if self.args.dataset == 'CREMAD' or self.args.dataset == 'AVE':
            model_len_a = process_grad(self.model.audio_net.parameters()).shape[0]
            model_len_v = process_grad(self.model.visual_net.parameters()).shape[0]
        elif self.args.dataset == 'CGMNIST':
            model_len_a = process_grad(self.model.gray_net.parameters()).shape[0]
            model_len_v = process_grad(self.model.colored_net.parameters()).shape[0]
        elif self.args.dataset == 'CrisisMMD':
            model_len_a = process_grad(self.model.imageEncoder.parameters()).shape[0]
            model_len_v = process_grad(self.model.textEncoder.parameters()).shape[0]
        else:
            raise ValueError('wrong dataset name.')
        global_grads = np.zeros(model_len_a+model_len_v)
        audio_grads = np.zeros(model_len_a)
        visual_grads = np.zeros(model_len_v)
        balance_ratios = 0

        if self.args.dataset != 'CrisisMMD':
            global_audio_pts = torch.zeros(self.n_classes, self.args.embed_dim).to(self.device)
            global_visual_pts = torch.zeros(self.n_classes, self.args.embed_dim).to(self.device)
        else:
            global_audio_pts = torch.zeros(self.n_classes, self.args.dim_img_repr).to(self.device)
            global_visual_pts = torch.zeros(self.n_classes, self.args.dim_text_repr).to(self.device)

        cc = 0
        samples = []

        for c_id in self.client_id_indices:
            print('cal the grads of client {}'.format(c_id))
            modality_type = self.get_client_type(c_id)
            num_samples, client_grads, ca_grads, cv_grads, b_ratio, audio_proto, visual_proto = self.trainer.get_PMR_grads(
                client_id=c_id,
                model=self.model,
                modality_type=modality_type,
                model_len=[model_len, model_len_a, model_len_v]
            )
            samples.append(num_samples)
            if cc == 0:
                intermediate_grads = np.zeros([len(self.client_id_indices) + 1, len(client_grads)])
                intermediate_grads_a = np.zeros([len(self.client_id_indices) + 1, len(ca_grads)])
                intermediate_grads_v = np.zeros([len(self.client_id_indices) + 1, len(cv_grads)])
                intermediate_ratios = []
                # audio_pts = []
                # visual_pts = []
            global_grads = np.add(global_grads, client_grads * num_samples)
            audio_grads = np.add(audio_grads, ca_grads * num_samples)  # 应该是乘上loss_proto对应的样本数量
            visual_grads = np.add(visual_grads, cv_grads * num_samples)
            balance_ratios += b_ratio * num_samples
            global_audio_pts = torch.add(global_audio_pts, audio_proto*num_samples)
            global_visual_pts = torch.add(global_visual_pts, visual_proto * num_samples)
            intermediate_grads[cc] = client_grads
            intermediate_grads_a[cc] = ca_grads
            intermediate_grads_v[cc] = cv_grads
            intermediate_ratios.append(b_ratio)
            # audio_pts.append(audio_proto)
            # visual_pts.append(visual_proto)
            cc += 1
        global_grads = global_grads * 1.0 / np.sum(np.asarray(samples))
        intermediate_grads[-1] = global_grads
        audio_grads = audio_grads * 1.0 / np.sum(np.asarray(samples))
        intermediate_grads_a[-1] = audio_grads
        visual_grads = visual_grads * 1.0 / np.sum(np.asarray(samples))
        intermediate_grads_v[-1] = visual_grads
        balance_ratios = balance_ratios * 1.0 / np.sum(np.asarray(samples))
        global_audio_pts = global_audio_pts * 1.0 / np.sum(np.asarray(samples))
        global_visual_pts = global_visual_pts * 1.0 / np.sum(np.asarray(samples))
        intermediate_ratios.append(balance_ratios)

        return intermediate_grads, intermediate_grads_a, intermediate_grads_v, np.asarray(
            intermediate_ratios), np.asarray(samples), global_audio_pts, global_visual_pts

    def get_client_type(self, client_id):
        if client_id in self.mm_client:
            modality_type = 'multimodal'
        elif client_id in self.audio_client:
            modality_type = 'audio'
        elif client_id in self.visual_client:
            modality_type = 'visual'
        else:
            raise ValueError('Non-exist modality type')
        return modality_type

    def select_cl_submod(self, round, clients_per_round=5, stochastic_greedy=True, balance_method=None):  # PMRsub
        if stochastic_greedy:
            SUi, using_modal = self.stochastic_greedy(clients_per_round, balance_method=balance_method)
        else:
            pass
        # print('Set Diff:', SUi0.difference(SUi), SUi.difference(SUi0))

        indices = np.array(list(SUi))
        selected_clients = np.asarray(self.client_id_indices)[indices]

        # return indices, selected_clients, gamma[indices]
        return selected_clients, using_modal

    def stochastic_greedy(self, num_clients, subsample=0.1, balance_method=None):
        print('global ratio: ', self.all_b_ratios[-1])
        # initialize the ground set and the selected set
        V_set = set(range(len(self.client_id_indices)))
        SUi = []
        using_modal = []

        m = max(num_clients, int(subsample * len(self.client_id_indices)))  # stochastic client batch
        for ni in range(num_clients):
            if m < len(V_set):
                R_set = np.random.choice(list(V_set), m, replace=False)  # select a larger subset from all clients
            else:
                R_set = list(V_set)
            # else:
            #     if m < len(V_set):
            #         R_set = np.random.choice(list(V_set), m, replace=False)  # select a larger subset from all clients
            #         Ra_set = np.random.choice(list(V_set), m, replace=False)  # select a larger subset from all clients
            #         Rv_set = np.random.choice(list(V_set), m, replace=False)  # select a larger subset from all clients
            #     else:
            #         R_set = Ra_set = Rv_set = list(V_set)
            if ni == 0:
                if balance_method is None:
                    marg_util = self.norm_diff[:, R_set].sum(0)
                    i = marg_util.argmin()
                    client_min = self.norm_diff[:, R_set[i]]
                elif balance_method == 'balansub' or balance_method == 'PMRsub':
                    # # method1: 矩阵相加
                    # global_b_ratio = self.all_b_ratios[-1]
                    # if global_b_ratio > self.args.balansubmod_thresh:  # 或是大于一定程度后再执行
                    #     norm_diff_fuse = self.norm_diff + global_b_ratio*self.norm_diff_v
                    #     marg_util = norm_diff_fuse[:, R_set].sum(0)
                    #     i = marg_util.argmin()
                    #     client_min = norm_diff_fuse[:, R_set[i]]

                    # method2: 矩阵分开计算，之后根据不均衡程度进行选择
                    global_b_ratio = self.all_b_ratios[-1]

                    marg_util = self.norm_diff[:, R_set].sum(0)
                    i = marg_util.argmin()
                    client_min = self.norm_diff[:, R_set[i]]
                    uni_visual = False
                    uni_audio = False
                    if global_b_ratio > 1:  # audio is dominant
                        marg_util_v = self.norm_diff_v[:, R_set].sum(0)
                        i_v = marg_util_v.argmin()
                        client_min_v = self.norm_diff_v[:, R_set[i_v]]

                        if self.all_b_ratios[R_set[i_v]] > self.args.balansubmod_thresh:
                            i = i_v
                            uni_visual = True
                            for cm in range(client_min.shape[0]):  # 选择了单模态数据，因此多模态client的这个值得重置
                                client_min[cm] = 100000000000
                        else:
                            for cm in range(client_min_v.shape[0]):  #
                                client_min_v[cm] = 100000000000

                    else:  # visual is dominant
                        marg_util_a = self.norm_diff_a[:, R_set].sum(0)
                        i_a = marg_util_a.argmin()
                        client_min_a = self.norm_diff_a[:, R_set[i_a]]

                        if self.all_b_ratios[R_set[i_a]] < 1/self.args.balansubmod_thresh:
                            i = i_a
                            uni_audio = True
                            for cm in range(client_min.shape[0]):  # 选择了单模态数据，因此多模态client的这个值得重置
                                client_min[cm] = 100000000000
                        else:
                            for cm in range(client_min_a.shape[0]):  #
                                client_min_a[cm] = 100000000000
                # elif balance_method == 'PMRsub':
                #     global_b_ratio = self.all_b_ratios[-1]
                #     marg_util = self.norm_diff[:, R_set].sum(0)
                #     i = marg_util.argmin()
                #     client_min = self.norm_diff[:, R_set[i]]
                #     uni_visual = False
                #     uni_audio = False
                #     if global_b_ratio > 1:  # audio is dominant
                #         marg_util_v = self.norm_diff_v[:, R_set].sum(0)
                #         i_v = marg_util_v.argmin()
                #         client_min_v = self.norm_diff_v[:, R_set[i_v]]
                #
                #         if self.all_b_ratios[R_set[i_v]] > self.args.balansubmod_thresh:
                #             i = i_v
                #             uni_visual = True
                #             for cm in range(client_min.shape[0]):  # 选择了单模态数据，因此多模态client的这个值得重置
                #                 client_min[cm] = 100000000
                #         else:
                #             for cm in range(client_min_v.shape[0]):  #
                #                 client_min_v[cm] = 100000000

            else:
                if balance_method is None:
                    client_min_R = np.minimum(client_min[:, None], self.norm_diff[:,R_set])
                    marg_util = client_min_R.sum(0)
                    i = marg_util.argmin()
                    client_min = client_min_R[:, i]
                elif balance_method == 'balansub' or balance_method == 'PMRsub':
                    global_b_ratio = self.all_b_ratios[-1]

                    client_min_R = np.minimum(client_min[:, None], self.norm_diff[:, R_set])
                    marg_util = client_min_R.sum(0)
                    i = marg_util.argmin()
                    # client_min = client_min_R[:, i]

                    uni_visual = False
                    uni_audio = False
                    if global_b_ratio > 1:  # audio is dominant
                        client_min_v_R = np.minimum(client_min_v[:, None], self.norm_diff_v[:, R_set])
                        marg_util_v = client_min_v_R.sum(0)
                        i_v = marg_util_v.argmin()
                        # client_min_v = client_min_v_R[:, i_v]

                        if self.all_b_ratios[R_set[i_v]] > self.args.balansubmod_thresh:
                            i = i_v
                            uni_visual = True
                            client_min_v = client_min_v_R[:, i_v]
                        else:
                            client_min = client_min_R[:, i]
                    else:
                        client_min_a_R = np.minimum(client_min_a[:, None], self.norm_diff_a[:, R_set])
                        marg_util_a = client_min_a_R.sum(0)
                        i_a = marg_util_a.argmin()
                        # client_min_v = client_min_v_R[:, i_v]

                        if self.all_b_ratios[R_set[i_a]] < 1/self.args.balansubmod_thresh:
                            i = i_a
                            uni_audio = True
                            client_min_a = client_min_a_R[:, i_a]
                        else:
                            client_min = client_min_R[:, i]
                # elif balance_method == 'PMRsub':
                #     global_b_ratio = self.all_b_ratios[-1]
                #
                #     client_min_R = np.minimum(client_min[:, None], self.norm_diff[:, R_set])
                #     marg_util = client_min_R.sum(0)
                #     i = marg_util.argmin()
                #     # client_min = client_min_R[:, i]
                #
                #     uni_visual = False
                #     uni_audio = False
                #     if global_b_ratio > 1:  # audio is dominant
                #         client_min_v_R = np.minimum(client_min_v[:, None], self.norm_diff_v[:, R_set])
                #         marg_util_v = client_min_v_R.sum(0)
                #         i_v = marg_util_v.argmin()
                #         # client_min_v = client_min_v_R[:, i_v]
                #
                #         if self.all_b_ratios[R_set[i_v]] > self.args.balansubmod_thresh:
                #             i = i_v
                #             uni_visual = True
                #             client_min_v = client_min_v_R[:, i_v]
                #         else:
                #             client_min = client_min_R[:, i]

            if balance_method is None:
                SUi.append(R_set[i])
                V_set.remove(R_set[i])
                using_modal.append('multi')
            else:
                SUi.append(R_set[i])
                V_set.remove(R_set[i])
                if uni_visual:
                    using_modal.append('visual')
                elif uni_audio:
                    using_modal.append('audio')
                else:
                    using_modal.append('multi')

        return SUi, using_modal
