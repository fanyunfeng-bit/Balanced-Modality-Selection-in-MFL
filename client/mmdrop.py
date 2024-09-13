from rich.console import Console
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader

from client.base import ClientBase
from config.utils import calculate_prototype, evaluate_modality_acc, process_grad, EU_dist, process_param_grad
import numpy as np


class FedMDropClient(ClientBase):
    def __init__(
        self,
        args,
        model,
        dataset: str,
        batch_size: int,
        local_epochs: int,
        local_lr: float,
        gpu: int,
    ):
        super(FedMDropClient, self).__init__(
            args,
            model,
            dataset,
            batch_size,
            local_epochs,
            local_lr,
            gpu,
        )

    def train(
            self,
            client_id: int,
            model,
            modality_type=None,
            global_epoch=0,
            global_proto=None
    ):
        self.global_epoch = global_epoch

        # update local lr
        if self.args.dataset != 'CrisisMMD':
            lr_epoch = 80
        else:
            lr_epoch = 25

        if self.global_epoch > lr_epoch:
            self.lr = self.args.local_lr - (self.args.local_lr - self.args.lr_min) / self.args.global_epochs * global_epoch
        else:
            self.lr = self.args.local_lr

        self.client_id = client_id
        self.modality_type = modality_type
        self.model.load_state_dict(model.state_dict())
        self.get_client_local_dataset()
        if self.client_id not in self.visited_client:
            self.visited_client.append(self.client_id)

        if global_proto is not None:
            self.global_audio_proto = global_proto[0]
            self.global_visual_proto = global_proto[1]

        all_params, data_num_client, [a_drop, v_drop], ratio = self._train()

        if self.args.clientsel_algo != 'PMR_submodular':
            if self.modality_type == 'multimodal':
                global_params = process_grad(model.parameters())
                local_params = process_grad(self.model.parameters())
            elif self.modality_type == 'audio':
                if self.args.dataset == 'ModelNet40':
                    global_params = process_grad(model.flow_net.parameters())
                    local_params = process_grad(self.model.flow_net.parameters())
                elif self.args.dataset != 'CGMNIST':
                    global_params = process_grad(model.audio_net.parameters())
                    local_params = process_grad(self.model.audio_net.parameters())
                else:
                    global_params = process_grad(model.gray_net.parameters())
                    local_params = process_grad(self.model.gray_net.parameters())
            elif self.modality_type == 'visual':
                if self.args.dataset != 'CGMNIST':
                    global_params = process_grad(model.visual_net.parameters())
                    local_params = process_grad(self.model.visual_net.parameters())
                else:
                    global_params = process_grad(model.colored_net.parameters())
                    local_params = process_grad(self.model.colored_net.parameters())
            else:
                raise ValueError('modality type error.')
        else:
            if self.modality_type == 'multimodal':
                if self.args.dataset == 'CREMAD' or self.args.dataset == 'AVE':
                    global_params = np.concatenate((process_grad(model.audio_net.parameters()), process_grad(model.visual_net.parameters())), 0)
                    local_params = np.concatenate((process_grad(self.model.audio_net.parameters()), process_grad(self.model.visual_net.parameters())), 0)
                elif self.args.dataset == 'CGMNIST':
                    global_params = np.concatenate(
                        (process_grad(model.gray_net.parameters()), process_grad(model.colored_net.parameters())), 0)
                    local_params = np.concatenate((process_grad(self.model.gray_net.parameters()),
                                                   process_grad(self.model.colored_net.parameters())), 0)
                elif self.args.dataset == 'CrisisMMD':
                    global_params = np.concatenate(
                        (process_grad(model.imageEncoder.parameters()), process_grad(model.textEncoder.parameters())), 0)
                    local_params = np.concatenate((process_grad(self.model.imageEncoder.parameters()),
                                                   process_grad(self.model.textEncoder.parameters())), 0)
                elif self.args.dataset == 'ModelNet40':
                    global_params = np.concatenate((process_grad(model.flow_net.parameters()), process_grad(model.visual_net.parameters())), 0)
                    local_params = np.concatenate((process_grad(self.model.flow_net.parameters()), process_grad(self.model.visual_net.parameters())), 0)

            elif self.modality_type == 'audio':
                if self.args.dataset == 'CREMAD' or self.args.dataset == 'AVE':
                    global_params = process_grad(model.audio_net.parameters())
                    local_params = process_grad(self.model.audio_net.parameters())
                elif self.args.dataset == 'CGMNIST':
                    global_params = process_grad(model.gray_net.parameters())
                    local_params = process_grad(self.model.gray_net.parameters())
                elif self.args.dataset == 'CrisisMMD':
                    global_params = process_grad(model.imageEncoder.parameters())
                    local_params = process_grad(self.model.imageEncoder.parameters())
                elif self.args.dataset == 'ModelNet40':
                    global_params = process_grad(model.flow_net.parameters())
                    local_params = process_grad(self.model.flow_net.parameters())

            elif self.modality_type == 'visual':
                if self.args.dataset == 'CREMAD' or self.args.dataset == 'AVE' or self.args.dataset == 'ModelNet40':
                    global_params = process_grad(model.visual_net.parameters())
                    local_params = process_grad(self.model.visual_net.parameters())
                elif self.args.dataset == 'CGMNIST':
                    global_params = process_grad(model.colored_net.parameters())
                    local_params = process_grad(self.model.colored_net.parameters())
                elif self.args.dataset == 'CrisisMMD':
                    global_params = process_grad(model.textEncoder.parameters())
                    local_params = process_grad(self.model.textEncoder.parameters())
            else:
                raise ValueError('modality type error.')

        client_grads = global_params - local_params

        return all_params, data_num_client, [a_drop, v_drop], client_grads, ratio

    def _train(self):
        optimizer: torch.optim.Optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.args.lr_decay_step, self.args.lr_decay_ratio)  # 基本不起调整lr的作用了，通过外部设定lr的大小

        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)

        train_dataloader = DataLoader(self.trainset, batch_size=self.batch_size,
                                      shuffle=True, pin_memory=False)  # 计算机的内存充足的时候，可以设置pin_memory=True

        # drop a modality or not based on local data
        if self.args.modality_drop:
            acc_a, acc_v = evaluate_modality_acc(self.args, self.model, train_dataloader, self.device, epoch=0, ratio=0.5)
            print(acc_a, acc_v)
            if acc_a / acc_v > self.args.drop_thresh:
                a_drop = True
                v_drop = False
            elif acc_v / acc_a > self.args.drop_thresh:
                v_drop = True
                a_drop = False
            else:
                a_drop = False
                v_drop = False
        else:
            a_drop = False
            v_drop = False

        # if self.args.dataset != 'CGMNIST':
        #     param_len_a = process_grad(self.model.audio_net.parameters()).shape[0]
        #     param_len_v = process_grad(self.model.visual_net.parameters()).shape[0]
        # else:
        #     param_len_a = process_grad(self.model.gray_net.parameters()).shape[0]
        #     param_len_v = process_grad(self.model.colored_net.parameters()).shape[0]
        # global_grads = np.zeros(param_len_a+param_len_v)
        # audio_grads = np.zeros(param_len_a)
        # visual_grads = np.zeros(param_len_v)

        # local training
        for le in range(self.local_epochs):
            self.model.train()
            _loss = 0
            if self.args.dataset == 'CrisisMMD':
                for step, data in enumerate(train_dataloader):
                    x = (data['image'].to(self.device),
                         {k: v.to(self.device) for k, v in data['text_tokens'].items()})
                    label = data['label'].to(self.device)

                    optimizer.zero_grad()

                    if self.modality_type == 'multimodal':
                        if a_drop is False and v_drop is False:
                            # TODO: make it simpler and easier to extend
                            a, v, out = self.model(x)
                            if self.args.clientsel_algo == 'PMR_submodular':
                                audio_sim = -EU_dist(a, self.global_audio_proto)  # B x n_class
                                visual_sim = -EU_dist(v, self.global_visual_proto)  # B x n_class
                        elif a_drop is True:
                            v, out = self.model.forward_text(x)
                            if self.args.clientsel_algo == 'PMR_submodular':
                                visual_sim = -EU_dist(v, self.global_visual_proto)  # B x n_class
                        elif v_drop is True:
                            a, out = self.model.forward_image(x)
                            if self.args.clientsel_algo == 'PMR_submodular':
                                audio_sim = -EU_dist(a, self.global_audio_proto)  # B x n_class
                        else:
                            raise ValueError('a_drop and v_drop cannot be True together!')
                    elif self.modality_type == 'audio':
                        a, out = self.model.forward_image(x)
                        if self.args.clientsel_algo == 'PMR_submodular':
                            audio_sim = -EU_dist(a, self.global_audio_proto)  # B x n_class
                        a_drop = False
                    elif self.modality_type == 'visual':
                        v, out = self.model.forward_text(x)
                        if self.args.clientsel_algo == 'PMR_submodular':
                            visual_sim = -EU_dist(v, self.global_visual_proto)  # B x n_class
                        v_drop = False
                    else:
                        raise ValueError('No such modality type.')

                    if self.args.clientsel_algo == 'PMR_submodular':
                        if self.modality_type == 'multimodal' and not a_drop and not v_drop:
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
                        _loss += loss.item()
                        optimizer.step()

                    else:
                        loss = criterion(out, label)
                        # print('loss: ', loss, loss_common)
                        _loss += loss.item()
                        loss.backward()
                        optimizer.step()
            else:
                for step, (spec, image, label) in enumerate(train_dataloader):
                    spec = spec.to(self.device)  # B x 257 x 1004(CREMAD 299)
                    image = image.to(self.device)  # B x 1(image count) x 3 x 224 x 224
                    label = label.to(self.device)  # B
                    B = label.shape[0]

                    optimizer.zero_grad()

                    if self.modality_type == 'multimodal':
                        if a_drop is False and v_drop is False:
                            # TODO: make it simpler and easier to extend
                            if self.args.dataset == 'ModelNet40':
                                a, v, out = self.model(spec, image, B)
                            elif self.args.dataset != 'CGMNIST':
                                a, v, out = self.model(spec.unsqueeze(1).float(), image.float())
                            else:
                                a, v, out = self.model(spec, image)  # gray colored
                            if self.args.clientsel_algo == 'PMR_submodular':
                                audio_sim = -EU_dist(a, self.global_audio_proto)  # B x n_class
                                visual_sim = -EU_dist(v, self.global_visual_proto)  # B x n_class
                        elif a_drop is True:
                            if self.args.dataset == 'ModelNet40':
                                v, out = self.model.forward_visual(image, B)
                            elif self.args.dataset != 'CGMNIST':
                                v, out = self.model.forward_visual(image.float(), bsz=label.shape[0])
                            else:
                                v, out = self.model.forward_colored(image)
                            if self.args.clientsel_algo == 'PMR_submodular':
                                visual_sim = -EU_dist(v, self.global_visual_proto)  # B x n_class
                        elif v_drop is True:
                            if self.args.dataset == 'ModelNet40':
                                a, out = self.model.forward_audio(spec, B)
                            elif self.args.dataset != 'CGMNIST':
                                a, out = self.model.forward_audio(spec.unsqueeze(1).float())
                            else:
                                a, out = self.model.forward_gray(spec)
                            if self.args.clientsel_algo == 'PMR_submodular':
                                audio_sim = -EU_dist(a, self.global_audio_proto)  # B x n_class
                        else:
                            raise ValueError('a_drop and v_drop cannot be True together!')
                    elif self.modality_type == 'audio':
                        if self.args.dataset == 'ModelNet40':
                            a, out = self.model.forward_audio(spec, B)
                        elif self.args.dataset != 'CGMNIST':
                            a, out = self.model.forward_audio(spec.unsqueeze(1).float())
                        else:
                            a, out = self.model.forward_gray(spec)
                        if self.args.clientsel_algo == 'PMR_submodular':
                            audio_sim = -EU_dist(a, self.global_audio_proto)  # B x n_class
                        a_drop = False
                    elif self.modality_type == 'visual':
                        if self.args.dataset == 'ModelNet40':
                            v, out = self.model.forward_visual(image, B)
                        elif self.args.dataset != 'CGMNIST':
                            v, out = self.model.forward_visual(image.float(), bsz=label.shape[0])
                        else:
                            v, out = self.model.forward_colored(image)
                        if self.args.clientsel_algo == 'PMR_submodular':
                            visual_sim = -EU_dist(v, self.global_visual_proto)  # B x n_class
                        v_drop = False
                    else:
                        raise ValueError('No such modality type.')

                    if self.args.clientsel_algo == 'PMR_submodular':
                        if self.modality_type == 'multimodal' and not a_drop and not v_drop:
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
                            # loss = loss + beta * loss_proto_a + lam * loss_proto_v
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
                        _loss += loss.item()
                        optimizer.step()

                    else:
                        loss = criterion(out, label)
                        # print('loss: ', loss, loss_common)
                        _loss += loss.item()
                        loss.backward()
                        optimizer.step()

            if self.args.optimizer == 'SGD':
                scheduler.step()

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
        elif self.args.dataset == 'ModelNet40':
            model_params = [self.model.flow_net.state_dict(), self.model.visual_net.state_dict(),
                            self.model.fusion_module.state_dict()]

        ratio = 1
        if self.args.clientsel_algo == 'balance_submodular':
            acc_a, acc_v = evaluate_modality_acc(self.args, self.model, train_dataloader, self.device, epoch=0,
                                                 ratio=0.5)
            ratio = acc_a / acc_v
        elif self.args.clientsel_algo == 'PMR_submodular':
            acc_a, acc_v, audio_proto, visual_proto = evaluate_modality_acc(self.args, self.model, train_dataloader,
                                                                            self.device, epoch=0,
                                                                            ratio=1.0, r_proto=True)
            ratio = acc_a / acc_v  # modality imbalance ratio
            self.audio_proto[self.client_id], self.visual_proto[self.client_id] = audio_proto, visual_proto
            # return model_params, self.trainset.data_num, [a_drop, v_drop], ratio

        return model_params, self.trainset.data_num, [a_drop, v_drop], ratio
