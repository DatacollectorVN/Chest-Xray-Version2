#https://towardsdatascience.com/how-to-work-with-object-detection-datasets-in-coco-format-9bf4fb5848a4
#https://cocodataset.org/?fbclid=IwAR1NaHHYmLfP4X2uVJ6zsUVTUP3G5RovpXW3Oor8AEdzkRgRhaPqnsd6Mxs#format-data
import pandas as pd 
from PIL import Image
from detectron2.structures import BoxMode 
import yaml
import json
import sys
sys.path.insert(1, "../src")
from utils_eda import *

FILE_TRAIN_CONFIG = os.path.join("..", "config", "streamlit_eda.yaml")
with open(FILE_TRAIN_CONFIG) as file:
    params = yaml.load(file, Loader = yaml.FullLoader)

def main():
    # dict()
    info_dict = {"description" : params["INFO_DESCRIPTION"],
                 "url" : params["INFO_URL"],
                 "version" : params["INFO_VERSION"],
                 "year" : params["INFO_YEAR"],
                 "contributor" : params["INFO_CONTRIBUTOR"]
                }
    
    # list(dict())
    lincenses = [{"url" : params["LIN_URL"],
                  "id" : params["LIN_ID"],
                  "name" : params["LIN_NAME"]
                 }
                ]

    # list(dict())
    categories = []
    for class_name in params["CLASSES_NAME"]:        
        category_dict =  {"supercategory" : class_name,
                          "id" : params["CLASSES_NAME"].index(class_name),
                          "name" : class_name,
                         }
        categories.append(category_dict)
    print(params["ANNOTATION_WBF_TRAIN_FILE"])
    df = pd.read_csv(params["ANNOTATION_WBF_TRAIN_FILE"])
    img_ids = df["image_file"].unique().tolist()
    imgs_list = [] # list(dict())
    annotations_list = [] # list(dict())
    bbox_id_cum = 0
    for i, img_id in tqdm(enumerate(img_ids), total = len(img_ids)):
        id = i + 1
        img_annotation = df[df["image_file"] == img_id]
        img = Image.open(os.path.join(params["IMG_DIR"], img_id))
        w, h = img.size
        img_dict = {"id" : id,
                    "license" : params["IMGS_LIN"],
                    "file_name" : img_id,
                    "coco_url" : params["IMGS_COCO_URL"],
                    "height" : h,
                    "width" : w,
                    "date_captured" : params["IMG_DATE_CAPTURED"],
                    "flickr_url" : params["IMG_FLICKR_URL"],
                   }
        imgs_list.append(img_dict)
        bboxes = img_annotation[["x_min", "y_min", "x_max", "y_max"]].values.tolist()
        bboxes = convert_bboxes_xyxy_to_xywh_COCO(bboxes)
        classes = img_annotation["class_name"].values.tolist()
        for j, bbox in enumerate(bboxes):
            bbox_id = bbox_id_cum + j + 1
            class_ = classes[j]
            area = bbox[2] * bbox[3]
            annotations = {"id" : bbox_id,
                           "image_id" : id,
                           "iscrowd" : 0, 
                           "category_id" : params["CLASSES_NAME"].index(class_),
                           "segmentation" : [],
                           "bbox" : bbox,
                           "area" : area,
                           "bbox_mode" : BoxMode.XYWH_ABS # XYWH_ABS = 1
                          }
            annotations_list.append(annotations)

            # reset value bbox_id_cum
            if j == len(bboxes) - 1:
                bbox_id_cum = bbox_id_cum + j + 1

    json_file = {"info" : info_dict,
                 "licenses" : lincenses,
                 "categories" : categories,
                 "images" : imgs_list,
                 "annotations" : annotations_list
                }
    with open(params["ANNOTATION_WBF_TRAIN_JSON_FILE"], 'w') as file:
        json.dump(json_file, file)
    
if __name__ == "__main__":
    main()
