import pandas as pd 
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer 
import yaml
import json
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
import os
import sys
sys.path.insert(1, "../src")
from utils_eda import *

FILE_TRAIN_CONFIG = os.path.join("..", "config", "streamlit_eda.yaml")
with open(FILE_TRAIN_CONFIG) as file:
    params = yaml.load(file, Loader = yaml.FullLoader)
with open(params["ANNOTATION_WBF_TRAIN_JSON_FILE"]) as file:
    json_file = json.load(file)
def main():
    register_coco_instances("chest_xray", metadata = {}, 
                            json_file = params["ANNOTATION_WBF_TRAIN_JSON_FILE"], 
                            image_root = params["IMG_DIR"])
    chest_xray_metadata = MetadataCatalog.get("chest_xray")
    print(json_file["images"][0]["file_name"])
    img_id = json_file["images"][0]["file_name"]
    img = cv2.imread(os.path.join(params["IMG_DIR"], img_id))
    visualizer = Visualizer(img[:, :, ::-1], metadata=chest_xray_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(json_file)
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.show()


if __name__ == "__main__":
    main()
