"""
@author: Originally by Francesco La Rosa
         Adapted by Vatsal Raina, Nataliia Molchanova
"""

import argparse
import os
import torch
from torch import nn
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.networks.nets import UNet, UNETR
import numpy as np
import random
from metrics import dice_metric
from data_load import get_train_dataloader, get_val_dataloader


