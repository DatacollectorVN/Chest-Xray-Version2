import pandas as pd
import numpy as np
import cv2
import os 
import sys
from tqdm import tqdm
import yaml
sys.path.insert(1, "../src")
from utils_eda import padding

FILE_NON_AUG = os.path.join("..", "config", "eda_non_aug.yaml")
with open(FILE_NON_AUG) as file:
    params_non_aug = yaml.load(file, Loader = yaml.FullLoader)

FILE_PADDING = os.path.join("..", "config", "eda_padding.yaml")
with open(FILE_PADDING) as file:
    params_padding = yaml.load(file, Loader = yaml.FullLoader)

def main():
    print(params_non_aug["ANNOTATIONS_VAL"])
    df = pd.read_csv(params_non_aug["ANNOTATIONS_VAL"], index_col = 0)
    img_ids = df["image_file"].unique().tolist()
    img_ids_lst = []
    classes_name_lst = []
    for i, img_id in tqdm(enumerate(img_ids), total = len(img_ids)):
        img = cv2.imread(os.path.join(params_non_aug["IMG_DIR"], img_id))
        df_img_id = df[df["image_file"] == img_id]
        bboxes = df_img_id[["x_min", "y_min", "x_max", "y_max"]].values.tolist()
        classes_name = df_img_id["class_name"].values.tolist()
        classes_name_lst.extend(classes_name)
        padding_img, padding_bboxes = padding(img, bboxes)
        cv2.imwrite(os.path.join(params_padding["IMG_DIR"], img_id), padding_img)
        img_id_lst = [img_id for _ in range(len(classes_name))]    
        img_ids_lst.extend(img_id_lst)
        if i == 0:
            rows = np.array(padding_bboxes)
        else:
            sub_row = np.array(padding_bboxes)
            rows = np.vstack([rows, sub_row])
    df_new = pd.DataFrame(rows, columns = ["x_min", "y_min", "x_max", "y_max"])
    df_new["image_file"] = img_ids_lst
    df_new["class_name"] = classes_name_lst
    df_new = df_new[["image_file", "class_name", "x_min", "y_min", "x_max", "y_max"]]
    df_new.to_csv(params_padding["ANNOTATIONS_VAL"], index = False)

if __name__ == "__main__":
    main()