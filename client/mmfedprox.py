from config.utils import trainable_params, common_loss
import torch
from rich.console import Console
from client.base import ClientBase
import torch.nn as nn
from torch.utils.data import Subset, DataLoader


class FedProxClient(ClientBase):
    def __init__(self,
                 args,
                 model: torch.nn.Module,
                 dataset: str,
                 batch_size: int,
                 local_epochs: int,
                 local_lr: float,
                 gpu: int, ):
        super(FedProxClient, self).__init__(
            args,
            model,
            dataset,
            batch_size,
            local_epochs,
            local_lr,
            gpu)
        self.mu = args.mu

    def _train(self):
        global_params = [p.clone().detach() for p in trainable_params(self.model)]

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
                spec = spec.to(self.device)  # B x 257 x 1004(CREMAD 299)
                image = image.to(self.device)  # B x 1(image count) x 3 x 224 x 224
                label = label.to(self.device)  # B
                B = label.shape[0]

                optimizer.zero_grad()

                loss_common = 0
                if self.modality_type == 'multimodal':
                    # TODO: make it simpler and easier to extend
                    if self.args.dataset == 'ModelNet40':
                        a, v, out = self.model(spec, image, B)
                    elif self.args.dataset != 'CGMNIST':
                        a, v, out = self.model(spec.unsqueeze(1).float(), image.float())
                    else:
                        a, v, out = self.model(spec, image)  # gray colored
                elif self.modality_type == 'audio':
                    if self.args.dataset == 'ModelNet40':
                        a, v, out = self.model.forward_audio(spec, B)
                    else:
                        a, out = self.model.forward_audio(spec.unsqueeze(1).float())
                elif self.modality_type == 'visual':
                    v, out = self.model.forward_visual(image.float(), bsz=label.shape[0])
                else:
                    raise ValueError('No such modality type.')

                loss = criterion(out, label) + loss_common
                # print('loss: ', loss, loss_common)
                _loss += loss.item()
                loss.backward()

                for w, w_t in zip(trainable_params(self.model), global_params):
                    if w.grad is not None:
                        w.grad.data += self.mu * (w.data - w_t.data)

                optimizer.step()

            if self.args.optimizer == 'SGD':
                scheduler.step()

            # if le % 10 == 0:
            #     acc = self.validate()
            #     print('acc in client {}: '.format(self.client_id), acc)

        print('client {} loss: '.format(self.client_id), _loss / len(train_dataloader))

        model_params = [self.model.audio_net.state_dict(), self.model.visual_net.state_dict(),
                        self.model.fusion_module.state_dict()]
        return model_params, self.trainset.data_num