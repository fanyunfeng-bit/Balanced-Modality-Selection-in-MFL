from rich.console import Console

from client.base import ClientBase


class FedAvgClient(ClientBase):
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
        super(FedAvgClient, self).__init__(
            args,
            model,
            dataset,
            batch_size,
            local_epochs,
            local_lr,
            gpu,
        )
