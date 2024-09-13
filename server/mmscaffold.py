import pickle
import random

import torch
from tqdm import tqdm

from server.base import ServerBase
from client.mmscaffold import SCAFFOLDClient
from config.utils import get_args
from copy import deepcopy


class SCAFFOLDServer(ServerBase):
    def __init__(self):
        super(SCAFFOLDServer, self).__init__(get_args(), "SCAFFOLD")

        self.trainer = SCAFFOLDClient(
            args=self.args,
            model=deepcopy(self.model),
            dataset=self.args.dataset,
            batch_size=self.args.batch_size,
            local_epochs=self.args.local_epochs,
            local_lr=self.args.local_lr,
            logger=self.logger,
            gpu=self.args.gpu,
        )

        # for client shift
        self.c_global_audio = [
            torch.zeros_like(param.data).to(self.device)
            for param in self.model.audio_net.parameters()
        ]
        self.c_global_visual = [
            torch.zeros_like(param.data).to(self.device)
            for param in self.model.visual_net.parameters()
        ]
        if self.args.fusion_method == 'sum':
            self.c_global_fusion_x = [
                torch.zeros_like(param.data).to(self.device)
                for param in self.model.fusion_module.fc_x.parameters()
            ]
            self.c_global_fusion_y = [
                torch.zeros_like(param.data).to(self.device)
                for param in self.model.fusion_module.fc_y.parameters()
            ]
        elif self.args.fusion_method == 'concat':
            self.c_global_fusion = [
                torch.zeros_like(param.data).to(self.device)
                for param in self.model.fusion_module.parameters()
            ]
        else:
            raise ValueError('error fusion method.')

        self.global_lr = 1.0
        self.training_acc = [[] for _ in range(self.global_epochs)]

    def train(self):

        progress_bar = tqdm(range(self.global_epochs), "Training...")

        f_log = open(self.log_file, 'a')

        for E in progress_bar:
            selected_clients = random.sample(
                self.client_id_indices, self.args.client_num_per_round
            )

            all_audio_params = []
            all_visual_params = []
            all_fusion_params = []
            all_client_data_num = []
            all_c_delta_audio = []
            all_c_delta_visual = []
            all_c_delta_fusion = []

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

                all_params, c_delta, data_num_client = self.trainer.train(
                    client_id=client_id,
                    model=deepcopy(self.model),
                    modality_type=modality_type,
                    c_global=[self.c_global_audio, self.c_global_visual, self.c_global_fusion],
                    verbose=(E % self.args.verbose_gap) == 0,
                )

                all_audio_params.append(deepcopy(all_params[0]))
                all_visual_params.append(deepcopy(all_params[1]))
                all_fusion_params.append(deepcopy(all_params[2]))
                all_client_data_num.append(data_num_client)
                all_c_delta_audio.append(deepcopy(c_delta[0]))
                all_c_delta_visual.append(deepcopy(c_delta[1]))
                all_c_delta_fusion.append(deepcopy(c_delta[2]))

            global_audio_weights, global_visual_weights, global_fusion_weights, audio_update, visual_update = \
                self.aggregate([all_audio_params, all_visual_params, all_fusion_params], [all_c_delta_audio, all_c_delta_visual, all_c_delta_fusion], all_client_data_num, selected_clients)

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

    def aggregate(self, all_client_params, all_c_delta, all_client_data_num, selected_clients):
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
        print('weights: ', weights, weights_audio, weights_visual)

        # update global model
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

        # update global control
        # print((self.args.client_num_per_round / len(self.client_id_indices)))
        for idx, c_g in enumerate(self.c_global_audio):
            for i in range(len(all_c_delta[0])):    # (len(self.mm_client) + len(self.audio_client))
                c_g.data += weights_audio[i] * all_c_delta[0][i][idx] * (self.args.client_num_per_round / len(self.client_id_indices))  # (self.args.client_num_per_round / len(self.client_id_indices))
        for idx, c_g in enumerate(self.c_global_visual):
            for i in range(len(all_c_delta[1])):
                c_g.data += weights_visual[i] * all_c_delta[1][i][idx] * (self.args.client_num_per_round / len(self.client_id_indices))
        for idx, c_g in enumerate(self.c_global_fusion):
            for i in range(len(all_c_delta[2])):
                c_g.data += weights[i] * all_c_delta[2][i][idx] * (self.args.client_num_per_round / len(self.client_id_indices))

        return w_audio_avg, w_visual_avg, w_fusion_avg, audio_update, visual_update


# if __name__ == "__main__":
#     server = SCAFFOLDServer()
#     server.run()


