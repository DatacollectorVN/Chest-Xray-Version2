# for verifying the outcome of get_chestxray_dicts
import numpy as np
import pandas as pd
import os 
import sys
import random
import cv2
import yaml
import matplotlib.pyplot as plt
from detectron2.data import MetadataCatalog, DatasetCatalog 
from detectron2.utils.visualizer import Visualizer
sys.path.insert(1, "../src")
from utils_eda import *

FILE_TRAIN_CONFIG = os.path.join("..", "config", "streamlit_eda.yaml")
RAD_LST = [f"R{i}" for i in range(1, 18)]
with open(FILE_TRAIN_CONFIG) as file:
    params = yaml.load(file, Loader = yaml.FullLoader)

def main():
    df = pd.read_csv(os.path.join(params["ANNOTATION_STANDARD_ADD_FILE"]))
    DatasetCatalog.register("my_dataset", lambda : get_chestxray_dicts(df, params["CLASSES_NAME"], params["IMG_DIR"]))
    MetadataCatalog.get("my_dataset").set(thing_classes = params["CLASSES_NAME"])
    chest_xray_metatdata = MetadataCatalog.get("my_dataset")
    dataset = get_chestxray_dicts(df, params["CLASSES_NAME"], params["IMG_DIR"])
    viz_imgs = []
    index_lst = [i for i in range(0, 200)] 
    for index in random.sample(index_lst, 6):
        sample = dataset[index]
        print(sample['file_name'].split('/')[-1])
        img = cv2.imread(sample["file_name"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        visualizer = Visualizer(img_rgb = img[:, :, ::-1], # img is loaded by opencv (BGR) --> ::-1 inverse channel tp (RGB)
                                metadata = chest_xray_metatdata, 
                                scale = 1)
        out = visualizer.draw_dataset_dict(dic = sample) # return VisImage object
        viz_imgs.append(out.get_image()[:, :, ::-1])
    
    plot_multi_imgs(viz_imgs, cols = 2, size = 20)
    plt.show()

if __name__ == "__main__":
    main()