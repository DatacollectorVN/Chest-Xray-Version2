import os
import shutil
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
    os.makedirs(params["IMG_DIR"], exist_ok = True)
    for img_id in tqdm(img_ids, total = len(img_ids)):
        source = os.path.join(params["IMG_DIR_ROOT"], img_id)
        dest = os.path.join(params["IMG_DIR"], img_id)
        shutil.copy(source, dest)

if __name__ == "__main__":
    main()