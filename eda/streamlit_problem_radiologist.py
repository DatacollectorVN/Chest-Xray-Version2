import streamlit as st
import pandas as pd 
import yaml
import os
import sys
sys.path.insert(1, "../src")
from utils_eda import *

FILE_TRAIN_CONFIG = os.path.join("..", "config", "eda_original.yaml")
RAD_LST = [f"R{i}" for i in range(1, 18)]
with open(FILE_TRAIN_CONFIG) as file:
    params = yaml.load(file, Loader = yaml.FullLoader)

# Basic UI
def main():
    st.header("**CHEST X-RAY VERSION2 PROBLEM FROM RADIOLOGIST**")
    st.write("by Kos Nhan")
    options_rad_id = RAD_LST
    options_rad_id.extend(["ALL"])
    rad_id_lst = st.sidebar.multiselect("Choose Radiologist ID, R1-7 just diagnose the negative classes", options = options_rad_id)
    options_classes_name = params["CLASSES_NAME"]
    options_classes_name.extend(["ALL"])
    classes_name_lst = st.sidebar.multiselect("Choose classes name", options = options_classes_name)
    mode = st.sidebar.selectbox("Choose mode", options = ["INDEX", "IMAGE'S FILE NAME"], index = 0)
    if mode == "INDEX":
        start_index, stop_index = st.sidebar.columns(2)
        start = start_index.number_input("None", min_value = 0, value = 0)
        stop = stop_index.number_input("None", min_value = 0, value = 1)
    else:
        img_file_name = st.sidebar.text_input("Enter image's file name", value = "NONE")

    if (rad_id_lst == []) or (classes_name_lst == []):
        return
    df = pd.read_csv(os.path.join(params["ANNOTATIONS_POSITIVE"]))
    img_ids_full = df["image_file"].unique().tolist()
    full_annotations = df.shape[0]
    full_images = len(img_ids_full)
    if "ALL" not in rad_id_lst:
        df = filter_rad_id(df, rad_id_lst = rad_id_lst)
    if "ALL" not in classes_name_lst:
        df = filter_classes_name(df, classes_name_lst = classes_name_lst)
    img_ids = df["image_file"].unique().tolist()
    st.write(f"**Number of annotations**= {df.shape[0]}/{full_annotations} with {len(img_ids)}/{full_images} images")
    if mode == "INDEX":
        st.dataframe(df)
        mode_index(df, img_ids, start, stop, params)
    else:
        if img_file_name == "NONE":
            st.write("Please enter image's file name")
        else:
            df = df[df["image_file"] == img_file_name]
            print(df)
            st.dataframe(df)
            mode_file_name(df, img_file_name, params)

if __name__ == "__main__":
    main()
   
    