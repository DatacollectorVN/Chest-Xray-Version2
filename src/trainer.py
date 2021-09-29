import pandas as pd 
from src.utils import get_fundus_dicts
from sklearn.model_selection import train_test_split
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, DatasetMapper
from detectron2.evaluation import COCOEvaluator
import detectron2.data.transforms as T
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os