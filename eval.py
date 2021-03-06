import os
import yaml
import gc
import torch
import sys
from src.custom_trainining_loop import do_test
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import inference_on_dataset
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import build_detection_test_loader
from detectron2.utils.logger import setup_logger
setup_logger()
import logging
logger = logging.getLogger("detectron2")
from src.custom_trainining_loop import get_evaluator

FILE_TRAIN_CONFIG = os.path.join("config", "inference.yaml")
with open(FILE_TRAIN_CONFIG) as file:
    params = yaml.load(file, Loader = yaml.FullLoader)

def setup_config_eval(params):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(params["MODEL"]))
    cfg.OUTPUT_DIR = params["OUTPUT_DIR"]
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, params["TRANSFER_LEARNING"])
     
    if "retina" in params["MODEL"]:
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = params["SCORE_THR_VAL"]
        cfg.MODEL.RETINANET.NUM_CLASSES = params["NUM_CLASSES"]
        cfg.MODEL.RETINANET.NMS_THRESH_TEST = params["NMS_THR_VAL"]
        cfg.MODEL.RETINANET.IOU_THRESHOLDS = params["RETINANET_IOU_THRESHOLDS"]
    else:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = params["SCORE_THR_VAL"]
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = params["NUM_CLASSES"]
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = params["NMS_THR_VAL"]
        cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = params["ROI_HEADS_IOU_THRESHOLDS"]
    
    cfg.DATASETS.TRAIN = (params["NAME_REGISTER"] + "train", )
    cfg.DATASETS.TEST = (params["NAME_REGISTER"] + "val", )
    cfg.INPUT.RANDOM_FLIP = params["RANDOM_FLIP"]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = params["BATCH_SIZE_PER_IMAGE"] 
    cfg.DATALOADER.NUM_WORKERS = params["NUM_WORKERS"]
    cfg.SOLVER.IMS_PER_BATCH = params["IMS_PER_BATCH"]
    
    return cfg

def main():
    register_coco_instances(params["NAME_REGISTER"] + "val", {}, 
                            params["ANNOTATION_VAL_JSON_FILE"], params["IMG_DIR"])
    register_coco_instances(params["NAME_REGISTER"] + "train", {}, 
                            params["ANNOTATION_TRAIN_JSON_FILE"], params["IMG_DIR"])
    cfg = setup_config_eval(params)
    logger.info(f"Use model {cfg.MODEL.WEIGHTS}")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)
    predictor = DefaultPredictor(cfg) # build model and load weight
    
    if params["CHECK_VAL"]:
        logger.info(f"Evaluation Validation dataset")
        do_test(cfg, predictor.model)
    
    if params["CHECK_TRAIN"]:
        logger.info(f"Evaluation Train dataset")
        for dataset_name in cfg.DATASETS.TRAIN:
            data_loader = build_detection_test_loader(cfg, dataset_name) # return torch.DataLoader
            evaluator = get_evaluator(cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference-train", dataset_name))
            inference_on_dataset(predictor.model, data_loader, evaluator)   
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:    
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
        print(f"clear memory")