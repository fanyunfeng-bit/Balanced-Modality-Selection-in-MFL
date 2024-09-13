import os
import pickle
import random
from collections import OrderedDict

import torch
from path import Path
from rich.console import Console
from rich.progress import track
from tqdm import tqdm

# sys.path.append('..')
from client.base import ClientBase
from config.utils import fix_random_seed, allocate_client_modality, EU_dist, dot_product_angle_tensor, \
    grad_amplitude_diff, calculate_prototype, process_grad, get_args
from data.utils.util import get_val_dataset
from models.basic_model import AVClassifier, DecomposedAVClassifier, CGClassifier, DenseNetBertMMModel
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from copy import deepcopy

from server.base import ServerBase
from client.pow_d import PowDClient

class PowDServer(ServerBase):
    def __init__(self):
        super(PowDServer, self).__init__(get_args(), "PowD")
        self.trainer = PowDClient(
            args=self.args,
            model=deepcopy(self.model),
            dataset=self.args.dataset,
            batch_size=self.args.batch_size,
            local_epochs=self.args.local_epochs,
            local_lr=self.args.local_lr,
            gpu=self.args.gpu,
        )
