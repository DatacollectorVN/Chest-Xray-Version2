import streamlit as st
import numpy as np 
import pandas as pd 
from PIL import Image
import yaml
import os
import shutil
import sys
import matplotlib.pyplot as plt
sys.path.insert(1, "../src")
from utils_eda import *

FILE_TRAIN_CONFIG = os.path.join("..", "config", "streamlit_eda.yaml")
RAD_LST = [f"R{i}" for i in range(1, 18)]
with open(FILE_TRAIN_CONFIG) as file:
    params = yaml.load(file, Loader = yaml.FullLoader)

def main():
    st.header("**CHEST X-RAY VERSION2 PROBLEM FROM RADIOLOGIST (HEATMAP)**")
    st.write("by Kos Nhan")
    options = ["ALL"]
    options.extend(RAD_LST)
    rad_id_lst = st.sidebar.multiselect("Choose Radiologist ID, R1-7 just diagnose the negative classes", options = options)
    if rad_id_lst == []:
        return
    df = pd.read_csv(os.path.join(params["ANNOTATION_STANDARD_ADD_FILE"]))
    img_ids_full = df["image_file"].unique().tolist()
    full_annotations = df.shape[0]
    full_images = len(img_ids_full)
    if "ALL" not in rad_id_lst:
        df = filter_rad_id(df, rad_id_lst = rad_id_lst)
    img_ids = df["image_file"].unique().tolist()
    st.write(f"**Number of annotations**= {df.shape[0]}/{full_annotations} with {len(img_ids)}/{full_images} images")
    df = get_bbox_norm(df)
    heatmap_size = (params["HEATMAP_SIZE"][0], params["HEATMAP_SIZE"][1])
    df["xmin_norm"] = df["xmin_norm"] * heatmap_size[1] # width
    df["ymin_norm"] = df["ymin_norm"] * heatmap_size[0] # height
    df["xmax_norm"] = df["xmax_norm"] * heatmap_size[1] # width
    df["ymax_norm"] = df["ymax_norm"] * heatmap_size[0] # height
    df[["xmin_norm", "ymin_norm", "xmax_norm", "ymax_norm"]] = df[["xmin_norm", "ymin_norm", "xmax_norm", "ymax_norm"]].astype(int)
    for row in range(7):
        globals()["col_1"], globals()["col_2"] = st.columns([1, 1])
        for col in range(2):
            index = (row * 2) + col
            plt.figure(figsize = (10,10))
            heatmap, num_bboxes, mean_bboxes_area_norm = draw_heatmap(df, params["CLASSES_NAME"][index], heatmap_size)
            plt.imshow(heatmap, cmap = "inferno", interpolation = 'nearest')
            os.makedirs("./outputs", exist_ok = True)
            if params["CLASSES_NAME"][index] != "Nodule/Mass":
                file_save = params["CLASSES_NAME"][index]
            else:
                file_save = "Nodule_Mass"
            plt.savefig(os.path.join("outputs", file_save + ".png"))
            img = Image.open(os.path.join("outputs", file_save + ".png")).convert("RGB")
            img = np.asarray(img)
            globals()[f"col_{col+1}"].write(f"**Class**: {file_save}")
            globals()[f"col_{col+1}"].write(f"**Number of bboxes**: {num_bboxes}")
            globals()[f"col_{col+1}"].write(f"**Mean of bboxes area norm**: {mean_bboxes_area_norm}")
            globals()[f"col_{col+1}"].image(img)

    shutil.rmtree("./outputs")
        
if __name__ == "__main__":
    main()
    