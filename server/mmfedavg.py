# import sys
# sys.path.append("../src")
from .base import ServerBase
from client.mmfedavg import FedAvgClient
from config.utils import get_args
from copy import deepcopy


class FedAvgServer(ServerBase):
    def __init__(self):
        super(FedAvgServer, self).__init__(get_args(), "FedAvg")
        self.trainer = FedAvgClient(
            args=self.args,
            model=deepcopy(self.model),
            dataset=self.args.dataset,
            batch_size=self.args.batch_size,
            local_epochs=self.args.local_epochs,
            local_lr=self.args.local_lr,
            gpu=self.args.gpu,
        )






