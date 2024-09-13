import os
import pickle
import random
from argparse import Namespace
from collections import OrderedDict

import torch
from path import Path
from rich.console import Console
from rich.progress import track
from tqdm import tqdm

from client.base import ClientBase
from config.utils import fix_random_seed, common_loss, calculate_prototype, EU_dist, process_grad, \
    evaluate_modality_acc, process_param_grad
from data.utils.util import get_train_dataset, get_val_dataset, CremadDataset
from models.basic_model import AVClassifier
from torch.utils.data import Subset, DataLoader
from copy import deepcopy
from typing import Dict, List, OrderedDict, Tuple
import numpy as np
import torch.nn as nn

label_key = 'label'


class PowDClient(ClientBase):
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
        super(PowDClient, self).__init__(
            args,
            model,
            dataset,
            batch_size,
            local_epochs,
            local_lr,
            gpu,
        )



