from argparse import Namespace
from copy import deepcopy

from config.utils import get_args
from .base import ServerBase
from client.mmfedprox import FedProxClient


class FedProxServer(ServerBase):
    def __init__(
        self,
        args: Namespace = None,
        algo: str = "FedProx",
        # unique_model=False,
        # default_trainer=False,
    ):
        if args is None:
            args = get_args()
        super().__init__(args, algo)
        self.trainer = FedProxClient(
            args=args,
            model=deepcopy(self.model),
            dataset=self.args.dataset,
            batch_size=self.args.batch_size,
            local_epochs=self.args.local_epochs,
            local_lr=self.args.local_lr,
            gpu=self.args.gpu,)


