# https://viblo.asia/p/state-of-the-art-instance-segmentation-chi-vai-dong-code-voi-detectron2-vyDZO7W9Zwj
import os
import yaml
import gc
import torch
import sys
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, DatasetMapper
from detectron2.data import build_detection_test_loader
from detectron2.config import get_cfg
from detectron2 import model_zoo

FILE_TRAIN_CONFIG = os.path.join("config", "train.yaml")
with open(FILE_TRAIN_CONFIG) as file:
    params = yaml.load(file, Loader = yaml.FullLoader)

def setup_config_test(params):
    #https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#update-the-config-for-new-datasets
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(params["MODEL"]))
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = params["BATCH_SIZE_PER_IMAGE"] 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = params["NUM_CLASSES"] 
    if "retina" in params["MODEL"]:
        cfg.MODEL.RETINANET.NUM_CLASSES = params["NUM_CLASSES"]
    cfg.DATASETS.TRAIN = (params["NAME_REGISTER"] + "train", )
    cfg.DATASETS.TEST = (params["NAME_REGISTER"] + "val", )
    cfg.DATALOADER.NUM_WORKERS = params["NUM_WORKERS"]
    cfg.SOLVER.IMS_PER_BATCH = params["IMS_PER_BATCH"]
    return cfg

def main():
    register_coco_instances(params["NAME_REGISTER"] + "train", {}, 
                            params["ANNOTATION_TRAIN_JSON_FILE"], params["IMG_DIR"])
    register_coco_instances(params["NAME_REGISTER"] + "val", {}, 
                            params["ANNOTATION_VAL_JSON_FILE"], params["IMG_DIR"])
    cfg = setup_config_test(params)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume = True)
    predictor = DefaultPredictor(cfg)
    meta_test = MetadataCatalog.get(params["NAME_REGISTER"] + "val")
    dicts_test = DatasetCatalog.get(params["NAME_REGISTER"] + "val")
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