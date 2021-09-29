import os 
import yaml
from src.trainer import FundusTrainer, COCOTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
import gc
import torch
import sys

FILE_TRAIN_CONFIG = os.path.join("config", "train.yaml")