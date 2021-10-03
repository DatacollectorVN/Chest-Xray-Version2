import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(1, "../src")
from utils_eda import *

FILE_TRAIN_CONFIG = os.path.join("..", "config", "streamlit_eda.yaml")
RAD_LST = [f"R{i}" for i in range(1, 18)]
with open(FILE_TRAIN_CONFIG) as file:
    params = yaml.load(file, Loader = yaml.FullLoader)

def main():
    df = pd.read_csv(params["ANNOTATION_WBF_FILE"])
    img_ids = df["image_file"].unique().tolist()
    imgs_train, imgs_val = train_test_split(img_ids, test_size = params["VAL_RATIO"], random_state = params["SEED"])
    df_train = filter_img_ids(df, imgs_train)
    df_val = filter_img_ids(df, imgs_val)
    df_train.to_csv(params["ANNOTATION_WBF_TRAIN_FILE"])
    df_val.to_csv(params["ANNOTATION_WBF_VAL_FILE"])
    print(f"Done save")
if __name__ == "__main__":
    main()
