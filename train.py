import os
import yaml
import gc
import torch
import sys
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2 import model_zoo
from src.custom_trainining_loop import *
from detectron2.utils.logger import setup_logger
setup_logger() # enable the logger. https://github.com/facebookresearch/detectron2/issues/144
import logging
logger = logging.getLogger("detectron2")

FILE_TRAIN_CONFIG = os.path.join("config", "train.yaml")
with open(FILE_TRAIN_CONFIG) as file:
    params = yaml.load(file, Loader = yaml.FullLoader)

def setup_config_train(params):
    #https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#update-the-config-for-new-datasets
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(params["MODEL"]))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(params["MODEL"])
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = params["BATCH_SIZE_PER_IMAGE"] 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = params["NUM_CLASSES"] 
    if "retina" in params["MODEL"]:
        cfg.MODEL.RETINANET.NUM_CLASSES = params["NUM_CLASSES"]
    cfg.DATASETS.TRAIN = (params["NAME_REGISTER"] + "train", )
    cfg.DATASETS.TEST = (params["NAME_REGISTER"] + "val", )
    cfg.DATALOADER.NUM_WORKERS = params["NUM_WORKERS"]
    cfg.SOLVER.IMS_PER_BATCH = params["IMS_PER_BATCH"]
    cfg.SOLVER.BASE_LR = params["BASE_LR"]
    cfg.SOLVER.WARMUP_ITERS = params["WARMUP_ITERS"]
    cfg.SOLVER.MAX_ITER = params["MAX_ITER"]
    cfg.SOLVER.STEPS = (params["STEPS_MIN"], params["STEPS_MAX"])
    cfg.SOLVER.GAMMA = params["GAMMA"]
    cfg.SOLVER.LR_SCHEDULER_NAME = params["LR_SCHEDULER_NAME"]
    cfg.INPUT.RANDOM_FLIP = params["RANDOM_FLIP"]
    #cfg.TEST.EVAL_PERIOD = params["EVAL_PERIOD"]
    return cfg

def main():
    register_coco_instances(params["NAME_REGISTER"] + "train", {}, 
                            params["ANNOTATION_TRAIN_JSON_FILE"], params["IMG_DIR"])
    register_coco_instances(params["NAME_REGISTER"] + "val", {}, 
                            params["ANNOTATION_VAL_JSON_FILE"], params["IMG_DIR"])
    cfg = setup_config_train(params)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)
    model = build_model(cfg) 
    do_train(cfg, model, False)
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:    
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
        print(f"clear memory")