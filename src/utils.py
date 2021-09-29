import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from collections import Counter
from ensemble_boxes import nms, weighted_boxes_fusion
from detectron2.structures import BoxMode
import os

def plot_multi_imgs(imgs, # 1 batchs contain multiple images
                    cols = 2, size = 10, # size of figure
                    is_rgb = True, title = None, cmap = "gray", save = False,
                    img_size = None): # set img_size if you want (width, height)
    rows = (len(imgs) // cols) + 1
    fig = plt.figure(figsize = (size *  cols, size * rows))
    for i , img in enumerate(imgs):
        if img_size is not None:
            img = cv2.resize(img, img_size)
        fig.add_subplot(rows, cols, i + 1) # add subplot int the the figure
        plt.imshow(img, cmap = cmap) # plot individual image
    plt.suptitle(title)
    if save:
        os.makedirs("outputs", exist_ok = True)
        plt.savefig(os.path.join("outputs", save))

def draw_bbox(img, img_bboxes, img_classes_name, classes_name, color, thickness=5):
    img_draw = img.copy()
    for i, img_bbox in enumerate(img_bboxes):
        img_draw = cv2.rectangle(img_draw, pt1 = (int(img_bbox[0]), int(img_bbox[1])), 
                                 pt2 = (int(img_bbox[2]), int(img_bbox[3])), 
                                 color = color[classes_name.index(img_classes_name[i])],
                                 thickness = thickness) 
        
        cv2.putText(img_draw,
                    text = img_classes_name[i].upper(),
                    org = (int(img_bbox[0]), int(img_bbox[1]) - 5),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 0.6,
                    color = (255, 255, 255),
                    thickness = 1, lineType = cv2.LINE_AA)    
    return img_draw

def xray_NMS(df, img_id, params):
    img = cv2.imread(os.path.join(params["IMG_DIR"], img_id))
    height, width = img.shape[:2]
    img_annotations = df[df["image_file"] == img_id]
    classes_name_full = img_annotations["class_name"].tolist()
    classes_id_full = [params["CLASSES_NAME"].index(i) for i in classes_name_full]
    classes_id_count = Counter(classes_id_full)
    classes_id_unique = [params["CLASSES_NAME"].index(i) for i in img_annotations["class_name"].unique().tolist()] 
    norm_scale = np.hstack([width, height, width, height])

    # prepare input data of nms function
    bboxes_lst = []
    scores_lst = [] # all scores is 1, cause all of bounding boxes is labeled by doctor
    classes_id_lst = []
    weights = [] # all weigths is 1, cause all of bounding boxes is ground truth 

    # have a list to save a single boxes cause the NMS function do not allow this case
    classes_id_single_lst = []
    bboxes_single_lst = []
    for class_id in classes_id_unique:
        if classes_id_count[class_id] == 1:
            classes_id_single_lst.append(class_id)
            bboxes_single_lst.append(img_annotations[img_annotations["class_name"] == params["CLASSES_NAME"][class_id]][["x_min", "y_min", "x_max", "y_max"]].to_numpy().squeeze().tolist())
        else:
            cls_id_lst = [class_id for _ in range(classes_id_count[class_id])]
            scores_ = np.ones(shape = classes_id_count[class_id]).tolist()
            bboxes = img_annotations[img_annotations["class_name"] == params["CLASSES_NAME"][class_id]][["x_min", "y_min", "x_max", "y_max"]].to_numpy()
            bboxes = bboxes / norm_scale
            bboxes = np.clip(bboxes, 0, 1).tolist()
            classes_id_lst.append(cls_id_lst)
            bboxes_lst.append(bboxes)
            scores_lst.append(scores_)
            weights.append(1)

    if classes_id_lst == []:
        boxes_nms = []
        classes_ids_nms = []
    else:
        boxes_nms, scores, classes_ids_nms = nms(boxes = bboxes_lst, scores = scores_lst, labels = classes_id_lst,
                                                 iou_thr = params["IOU_THR_NMS"], weights = weights)
        boxes_nms = boxes_nms * norm_scale
        boxes_nms = boxes_nms.astype(int).tolist()
        classes_ids_nms = classes_ids_nms.astype(int).tolist()

    # add with single class
    boxes_nms.extend(bboxes_single_lst)
    classes_ids_nms.extend(classes_id_single_lst)
    classes_name_nms = [params["CLASSES_NAME"][j] for j in classes_ids_nms]
    return boxes_nms, classes_name_nms

def xray_WBF(df, img_id, params):
    img = cv2.imread(os.path.join(params["IMG_DIR"], img_id))
    height, width = img.shape[:2]
    img_annotations = df[df["image_file"] == img_id]
    classes_name_full = img_annotations["class_name"].tolist()
    classes_id_full = [params["CLASSES_NAME"].index(i) for i in classes_name_full]
    classes_id_count = Counter(classes_id_full)
    classes_id_unique = [params["CLASSES_NAME"].index(i) for i in img_annotations["class_name"].unique().tolist()] 
    norm_scale = np.hstack([width, height, width, height])

    # prepare input data of nms function
    bboxes_lst = []
    scores_lst = [] # all scores is 1, cause all of bounding boxes is labeled by doctor
    classes_id_lst = []
    weights = [] # all weigths is 1, cause all of bounding boxes is ground truth 

    # have a list to save a single boxes cause the NMS function do not allow this case
    classes_id_single_lst = []
    bboxes_single_lst = []
    for class_id in classes_id_unique:
        if classes_id_count[class_id] == 1:
            classes_id_single_lst.append(class_id)
            bboxes_single_lst.append(img_annotations[img_annotations["class_name"] == params["CLASSES_NAME"][class_id]][["x_min", "y_min", "x_max", "y_max"]].to_numpy().squeeze().tolist())
        else:
            cls_id_lst = [class_id for _ in range(classes_id_count[class_id])]
            scores_ = np.ones(shape = classes_id_count[class_id]).tolist()
            bboxes = img_annotations[img_annotations["class_name"] == params["CLASSES_NAME"][class_id]][["x_min", "y_min", "x_max", "y_max"]].to_numpy()
            bboxes = bboxes / norm_scale
            bboxes = np.clip(bboxes, 0, 1).tolist()
            classes_id_lst.append(cls_id_lst)
            bboxes_lst.append(bboxes)
            scores_lst.append(scores_)
            weights.append(1)
    if classes_id_lst == []:
        boxes_nms = []
        classes_ids_nms = []
    else:
        boxes_nms, scores, classes_ids_nms = weighted_boxes_fusion(boxes_list = bboxes_lst, scores_list = scores_lst, labels_list = classes_id_lst,
                                                                   iou_thr = params["IOU_THR_WBF"], weights = weights, skip_box_thr = params["SKIP_BOX_THR_WBF"])
        boxes_nms = boxes_nms * norm_scale
        boxes_nms = boxes_nms.astype(int).tolist()
        classes_ids_nms = classes_ids_nms.astype(int).tolist()

    # add with single class
    boxes_nms.extend(bboxes_single_lst)
    classes_ids_nms.extend(classes_id_single_lst)
    classes_name_nms = [params["CLASSES_NAME"][j] for j in classes_ids_nms]
    return boxes_nms, classes_name_nms

def x_ray_train_val_split(df, val_ratio):
    # problem 2 in standard data
    train_standard = ["ec513a0af055499f1b188cc6a9175ee1.jpg", 
                      "f9f7feefb4bac748ff7ad313e4a78906.jpg"]
    val_standard = ["43e11813c6d7bcef779a1a287edc02c4.jpg"]