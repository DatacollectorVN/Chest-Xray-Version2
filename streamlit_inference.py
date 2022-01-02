import os
import yaml
import json
from PIL import Image
import cv2
import numpy as np
from src.utils import detectron2_prediction, get_outputs_detectron2, draw_bbox_infer
from detectron2.engine import  DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import streamlit as st
import time
from detectron2.utils.logger import setup_logger
setup_logger()
import logging
logger = logging.getLogger("detectron2")

FILE_INFER_CONFIG = os.path.join("config", "inference.yaml")
with open(FILE_INFER_CONFIG) as file:
    params = yaml.load(file, Loader = yaml.FullLoader)

def setup_config_infer(params):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(params["MODEL"]))
    cfg.OUTPUT_DIR = params["OUTPUT_DIR"]
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, params["TRANSFER_LEARNING"])
    cfg.DATALOADER.NUM_WORKERS = 0
    
    if "retina" in params["MODEL"]:
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = params["SCORE_THR"]
        cfg.MODEL.RETINANET.NUM_CLASSES = params["NUM_CLASSES"]
        cfg.MODEL.RETINANET.NMS_THRESH_TEST = params["NMS_THR"]
    else:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = params["SCORE_THR"]
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = params["NUM_CLASSES"]
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = params["NMS_THR"]

    return cfg

def main():
    # UI of website
    st.header("**CHEST X-RAY VERSION2 PROBLEM FROM RADIOLOGIST**")
    st.write("by Kos Nhan")
    file = st.sidebar.file_uploader('Upload img file (JPG/PNG format)')
    params["SCORE_THR"] = st.sidebar.number_input("Confidence Score Threshold", min_value = 0.0, max_value = 11.0, format = "%f", value = 0.5)
    params["NMS_THR"] = st.sidebar.number_input("IOU NMS Threshold", min_value = 0.0, max_value = 1.0, format = "%f", value = 0.5, )
    if not file:
        st.write("Please upload your image (png, jpg)")
        return
    
    cfg = setup_config_infer(params)
    model = DefaultPredictor(cfg)
    img = Image.open(file)
    img = np.array(img.convert("RGB"))
    st.image(img, caption = "Orginal image")

    start = time.time()
    outputs = detectron2_prediction(model, img)
    duration = time.time() - start
    st.write(f"Duration: {duration}")
    pred_bboxes, pred_confidence_scores, pred_classes = get_outputs_detectron2(outputs)
    pred_bboxes = pred_bboxes.detach().numpy().astype(int)
    pred_confidence_scores = pred_confidence_scores.detach().numpy()
    pred_confidence_scores = np.round(pred_confidence_scores, 2)
    pred_classes = pred_classes.detach().numpy().astype(int)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_after = draw_bbox_infer(img, pred_bboxes, 
                                pred_classes, pred_confidence_scores,
                                params["CLASSES_NAME"], params["COLOR"], 5)
    st.image(img_after, caption = "Image after prediction")

if __name__ == "__main__":
    main()