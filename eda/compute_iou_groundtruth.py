import pandas as pd 
import os
import sys
import yaml
from itertools import combinations
from tqdm import tqdm
sys.path.insert(1, "../src")
from utils_eda import *

FILE_TRAIN_CONFIG = os.path.join("..", "config", "streamlit_eda.yaml")
RAD_LST = [f"R{i}" for i in range(1, 18)]
with open(FILE_TRAIN_CONFIG) as file:
    params = yaml.load(file, Loader = yaml.FullLoader)

def main():
    df = pd.read_csv(os.path.join(params["ANNOTATION_STANDARD_ADD_FILE"]))
    for class_name in params["CLASSES_NAME"]:
        df_sub = df[df["class_name"] == class_name]
        img_ids = df_sub["image_file"].unique().tolist()
        iou_mean_all_imgs = 0
        img_with_single_classes = 0
        print(f"class_name = {class_name}")
        for img_id in tqdm(img_ids, total = len(img_ids)):
            bboxes = df_sub[df_sub["image_file"] == img_id][["x_min", "y_min", "x_max", "y_max"]].to_numpy().tolist()
            combination_bboxes = list(combinations(bboxes, 2))
            if len(combination_bboxes) == 0:
                img_with_single_classes += 1
            else:               
                iou_sum_per_img = 0
                count = 0
                for pair_bboxes in combination_bboxes:
                    iou = compute_iou(bbox_1 = pair_bboxes[0], bbox_2 = pair_bboxes[1])                   
                    if 0 < iou <=1:
                        iou_sum_per_img += iou
                    else:
                        count +=1 
                if len(combination_bboxes) - count == 0:
                    img_with_single_classes +=1
                else: 
                    iou_mean_per_img = iou_sum_per_img / (len(combination_bboxes) - count)
                    iou_mean_all_imgs += iou_mean_per_img             
        iou_mean_per_class = iou_mean_all_imgs / (len(img_ids) - img_with_single_classes)
        print(f"class_name {class_name} with iou {iou_mean_per_class}")

if __name__ == "__main__":
    main()