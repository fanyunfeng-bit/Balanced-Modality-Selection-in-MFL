from rich.console import Console

from client.base import ClientBase


class FedMSplitClient(ClientBase):
    def __init__(
        self,
        args,
        model,
        dataset: str,
        batch_size: int,
        local_epochs: int,
        local_lr: float,
        logger: Console,
        gpu: int,
    ):
        super(FedMSplitClient, self).__init__(
            args,
            model,
            dataset,
            batch_size,
            local_epochs,
            local_lr,
            logger,
            gpu,
        )
