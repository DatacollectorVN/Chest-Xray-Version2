import streamlit as st
import pandas as pd 
import yaml
import os
import sys
sys.path.insert(1, "../src")
from utils_eda import *

FILE_TRAIN_CONFIG = os.path.join("..", "config", "eda_non_aug.yaml")
with open(FILE_TRAIN_CONFIG) as file:
        params = yaml.load(file, Loader = yaml.FullLoader)

# BASIC UI
def main():
    st.header("**CHEST X-RAY VERSION2 NMS TECHNIQUES**")
    st.write("by Kos Nhan")
    technique = st.sidebar.selectbox("Choose the technique:", options = ["NONE", "NMS", "WBF"], index = 0)
    options_classes_name = params["CLASSES_NAME"]
    options_classes_name.extend(["ALL"])
    classes_name_lst = st.sidebar.multiselect("Choose classes name", options = options_classes_name)
    if technique == "WBF":
        skip_box_thr = st.sidebar.number_input("skip boxes threshold", min_value = 0.0, max_value = 1.0, value = 0.1)
    else:
        skip_box_thr = None 
    iou_thr = st.sidebar.number_input("IOU threshold", min_value = 0.0, max_value = 1.0, value = 0.5)
    start_index, stop_index = st.sidebar.columns(2)
    start = start_index.number_input("None", min_value = 0, value = 0)
    stop = stop_index.number_input("None", min_value = 0, value = 1)
    if (technique == "NONE") or (classes_name_lst == []):
        return 
    df = pd.read_csv(os.path.join(params["ANNOTATIONS_PREPROCESS_BEFORE_WBF"]))
    if "ALL" not in classes_name_lst:
        df = filter_classes_name(df, classes_name_lst = classes_name_lst)
    img_ids = df["image_file"].unique().tolist()
    if (start ==0) and (stop ==0):
        st.write("Choose the values")
    elif start > stop:
        st.write("The value of start index must smaller than stop index")
    else:
        for i, img_id in enumerate(img_ids[start :stop]):
            img, img_after, meta_info = nms_wbf(df, img_id, params, technique, iou_thr, skip_box_thr)
            col_1, col_2 = st.columns([1, 1])
            if i == 0:
                col_1.write("***Image original***")
                col_2.write("***Image after***")
            col_1.write(f"**Image**: {img_id}")
            col_1.write(f"**Number of unique classes**: {meta_info['number_unique_classes']}")
            col_1.write(f"**All bboxes**: {meta_info['number_bboxes']}")
            col_2.write(f"**Index**: {i + start}")
            col_2.write(f"**Number of unique classes**: {meta_info['number_unique_classes_after']}")
            col_2.write(f"**All bboxes**: {meta_info['number_bboxes_after']}")            
            col_1.image(img)
            col_2.image(img_after)
        
if __name__ == "__main__":
    main()
