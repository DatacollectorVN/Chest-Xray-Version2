import os
import yaml
import gc
import torch
import sys
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from src.custom_trainining_loop import *
from detectron2.utils.logger import setup_logger
setup_logger()
import logging
logger = logging.getLogger("detectron2")

FILE_TRAIN_CONFIG = os.path.join("config", "inference.yaml")
with open(FILE_TRAIN_CONFIG) as file:
    params = yaml.load(file, Loader = yaml.FullLoader)

def setup_config_eval(params):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(params["MODEL"]))
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, params["TRANSFERLEARNING"])
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = params["SCORE_THRESH_TEST"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = params["NUM_CLASSES"] 
    if "retina" in params["MODEL"]:
        cfg.MODEL.RETINANET.NUM_CLASSES = params["NUM_CLASSES"]
    cfg.DATASETS.TRAIN = (params["NAME_REGISTER"] + "train", )
    cfg.DATASETS.TEST = (params["NAME_REGISTER"] + "val", )
    cfg.DATALOADER.NUM_WORKERS = params["NUM_WORKERS"]
    return cfg

def main():
    register_coco_instances(params["NAME_REGISTER"] + "train", {}, 
                            params["ANNOTATION_TRAIN_JSON_FILE"], params["IMG_DIR"])
    register_coco_instances(params["NAME_REGISTER"] + "val", {}, 
                            params["ANNOTATION_VAL_JSON_FILE"], params["IMG_DIR"])
    cfg = setup_config_eval(params)
    logger.info(f"Use model {cfg.MODEL.WEIGHTS}")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)
    ''' ERROR: mAP = 0 for all classes
    model = build_model(cfg)
    do_test(cfg, model)
    '''
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(True) 
    evaluator = COCOEvaluator(params["NAME_REGISTER"] + "val", cfg, False, output_dir="./output/")
    test_loader = build_detection_test_loader(cfg, params["NAME_REGISTER"] + "val")
    inference_on_dataset(trainer.model, test_loader, evaluator)
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:    
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
        print(f"clear memory")