import numpy as np 
import pandas as pd 
import os
import yaml
from tqdm import tqdm
import sys
sys.path.insert(1, "../src") # prevent the ImportError. https://stackoverflow.com/questions/4383571/importing-files-from-different-folder 
from utils import xray_NMS
BASE = ".."
FILE_TRAIN_CONFIG = os.path.join("..", "config", "streamlit_eda.yaml")

def main():
    with open(FILE_TRAIN_CONFIG) as file:
        params = yaml.load(file, Loader = yaml.FullLoader)
    df = pd.read_csv(os.path.join(params["ANNOTATION_STANDARD_ADD_FILE"]))
    img_ids = df["image_file"].unique().tolist()
    for i, img_id in tqdm(enumerate(img_ids), total = len(img_ids)):
        bboxes_nms, classes_name_nms = xray_NMS(df, img_id, params)
        for j in range(len(classes_name_nms)):
            if (i == 0) and (j == 0):
                row = []
                row.append(img_id)
                row.append(classes_name_nms[j])
                row.extend(bboxes_nms[j])
            else:
                row_sub = []
                row_sub.append(img_id)
                row_sub.append(classes_name_nms[j])
                row_sub.extend(bboxes_nms[j])
                row = np.vstack((row, row_sub))
    df_nms = pd.DataFrame(row, columns = ["image_file", "class_name", "x_min", "y_min", "x_max", "y_max"])
    df_nms.to_csv(params["ANNOTATION_NMS_FILE"], index = False)
    print("DONE")

if __name__ == "__main__":
    main()
