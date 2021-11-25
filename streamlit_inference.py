import streamlit as st
import pandas as pd 
import yaml
import os
import cv2
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

FILE_TRAIN_CONFIG = os.path.join("config", "inference.yaml")
with open(FILE_TRAIN_CONFIG) as file:
        params = yaml.load(file, Loader = yaml.FullLoader)

def setup_config_infer(params):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(params["MODEL"]))
    cfg.MODEL.WEIGHTS = os.path.join("runs", params["TRANSFERLEARNING"])
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # set threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = params["NUM_CLASSES"] 
    if "retina" in params["MODEL"]:
        cfg.MODEL.RETINANET.NUM_CLASSES = params["NUM_CLASSES"]
    cfg.DATASETS.TEST = (params["NAME_REGISTER"] + "val", )
    return cfg

def main():
    register_coco_instances(params["NAME_REGISTER"] + "val", {}, 
                            params["ANNOTATION_VAL_JSON_FILE"], params["IMG_DIR"])
    test_metadata = MetadataCatalog.get(params["NAME_REGISTER"] + "val")
    cfg = setup_config_infer(params)
    print(cfg.MODEL.WEIGHTS)
    predictor = DefaultPredictor(cfg)
    img = cv2.imread(os.path.join(params["IMG_DIR"], "ff6b2b10b8f2350ab09d8690c9c83154.jpg"))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    outputs = predictor(img)
    print(outputs)
    #st.image(out)

if __name__ == "__main__":
    main()