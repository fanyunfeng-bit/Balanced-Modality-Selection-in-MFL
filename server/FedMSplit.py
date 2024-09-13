# import sys
# sys.path.append("../src")
from .base import ServerBase
from client.FedMSplit import FedMSplitClient
from config.utils import get_args, fix_random_seed
from copy import deepcopy
from tqdm import tqdm
import random
import torch


class FedMSplitServer(ServerBase):
    def __init__(self):
        super(FedMSplitServer, self).__init__(get_args(), "FedMSplit")
        fix_random_seed(self.args.seed)

        self.all_models = [deepcopy(self.model) for _ in range(self.args.client_num)]

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
            modality_types = []

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

                self.trainer = FedMSplitClient(
                    args=self.args,
                    model=deepcopy(self.all_models[client_id]),
                    dataset=self.args.dataset,
                    batch_size=self.args.batch_size,
                    local_epochs=self.args.local_epochs,
                    local_lr=self.args.local_lr,
                    logger=self.logger,
                    gpu=self.args.gpu,
                )

                all_params, data_num_client = self.trainer.train(
                    client_id=client_id,
                    model=deepcopy(self.model),
                    modality_type=modality_type,
                    global_epoch=E,
                    global_proto=None
                )

                all_audio_params.append(deepcopy(all_params[0]))
                all_visual_params.append(deepcopy(all_params[1]))
                all_fusion_params.append(deepcopy(all_params[2]))
                all_client_data_num.append(data_num_client)
                modality_types.append[modality_type]

                # aggregation
            global_audio_weights, global_visual_weights, global_fusion_weights, audio_update, visual_update = \
                self.aggregate([all_audio_params, all_visual_params, all_fusion_params], all_client_data_num,
                               selected_clients)
            if audio_update:
                self.model.audio_net.load_state_dict(global_audio_weights)
            if visual_update:
                self.model.visual_net.load_state_dict(global_visual_weights)
            self.model.fusion_module.load_state_dict(global_fusion_weights)

    # @torch.no_grad()
    # def neighborhood_attentive_model_aggregation(self, all_client_params, all_client_data_num, selected_clients, modality_types):
    #     cal_model_sim_matrix(all_client_params, modality_types)
    #     all_client_params[0][0]
    #
    # def cal_model_sim_matrix(self, all_client_params, modality_types):






