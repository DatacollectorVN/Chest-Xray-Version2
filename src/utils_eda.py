import numpy as np
import cv2
import streamlit as st
import os 
from ensemble_boxes import nms, weighted_boxes_fusion
from collections import Counter

def draw_bbox_rad(img, img_bboxes, img_classes_name, classes_name, rad_id, color, thickness=5):
    img_draw = img.copy()
    for i, img_bbox in enumerate(img_bboxes):
        img_draw = cv2.rectangle(img_draw, pt1 = (int(img_bbox[0]), int(img_bbox[1])), 
                                 pt2 = (int(img_bbox[2]), int(img_bbox[3])), 
                                 color = color[classes_name.index(img_classes_name[i])],
                                 thickness = thickness) 
        cv2.putText(img_draw,
                    text = img_classes_name[i].upper() + "-" + rad_id[i].upper(),
                    org = (int(img_bbox[0]), int(img_bbox[1]) - 5),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 0.6,
                    color = (255, 0, 0),
                    thickness = 1, lineType = cv2.LINE_AA)     
    return img_draw

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
                    color = (255, 0, 0),
                    thickness = 1, lineType = cv2.LINE_AA)     
    return img_draw

def filter_rad_id(df, rad_id_lst):
    str_condition = ""
    for i, rad_id in enumerate(rad_id_lst):
        if i != len(rad_id_lst) - 1:
            str_condition += f"rad_id == '{rad_id}'|"
        else:
            str_condition += f"rad_id == '{rad_id}'"
    df = df[df.eval(str_condition)]
    return df

def filter_classes_name(df, classes_name_lst):
    str_condition = ""
    for i, class_name in enumerate(classes_name_lst):
        if i != len(classes_name_lst) - 1:
            str_condition += f"class_name == '{class_name}'|"
        else:
            str_condition += f"class_name == '{class_name}'"
    df = df[df.eval(str_condition)]
    return df

def mode_index(df, img_ids, start, stop, params):
    if (start ==0) and (stop ==0):
        st.write("Choose the values")
    elif start > stop:
        st.write("The value of start index must smaller than stop index")
    else:
        for i, img_id in enumerate(img_ids[start :stop]):
            img = cv2.imread(os.path.join(params["IMG_DIR"], img_id))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            df_img_id = df[df["image_file"] == img_id]
            bboxes = df_img_id[["x_min", "y_min", "x_max", "y_max"]].values.tolist()
            classes_name = df_img_id["class_name"].values.tolist()
            rad_id = df_img_id["rad_id"].values.tolist()
            classes_name_unique = np.unique(np.array(classes_name))
            rad_id_unique = np.unique(np.array(rad_id))
            st.write(f"**Image**: {img_id}")
            st.write(f"**Index**: {i + start}")
            st.write(f"**All bboxes**: {len(classes_name)}")
            st.write(f"**Number of unique classes**: {len(classes_name_unique)}")
            st.write(f"**Number of unique rad_id**: {len(rad_id_unique)}")
            img = draw_bbox_rad(img, bboxes, classes_name, params["CLASSES_NAME"], rad_id, params["COLOR"])
            st.image(img)

def mode_file_name(df, img_file_name, params):
    img = cv2.imread(os.path.join(params["IMG_DIR"], img_file_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    df_img_id = df[df["image_file"] == img_file_name]
    bboxes = df_img_id[["x_min", "y_min", "x_max", "y_max"]].values.tolist()
    classes_name = df_img_id["class_name"].values.tolist()
    rad_id = df_img_id["rad_id"].values.tolist()
    classes_name_unique = np.unique(np.array(classes_name))
    rad_id_unique = np.unique(np.array(rad_id))
    st.write(f"**Image**: {img_file_name}")
    st.write(f"**All bboxes**: {len(classes_name)}")
    st.write(f"**Number of unique classes**: {len(classes_name_unique)}")
    st.write(f"**Number of unique rad_id**: {len(rad_id_unique)}")
    img = draw_bbox_rad(img, bboxes, classes_name, params["CLASSES_NAME"], rad_id, params["COLOR"])
    st.image(img)

def get_bbox_norm(df):
    df["xmin_norm"] = df["x_min"] / df["image_width"]
    df["ymin_norm"] = df["y_min"] / df["image_height"]
    df["xmax_norm"] = df["x_max"] / df["image_width"]
    df["ymax_norm"] = df["y_max"] / df["image_height"] 
    return df

def get_mean_bboxes_area_norm(df, heatmap_size):
    mean_xmin_norm = df["xmin_norm"].mean() / heatmap_size[1]
    mean_ymin_norm = df["ymin_norm"].mean() / heatmap_size[0]
    mean_xmax_norm = df["xmax_norm"].mean() / heatmap_size[1]
    mean_ymax_norm = df["ymax_norm"].mean() / heatmap_size[0]
    return round((mean_xmax_norm - mean_xmin_norm) * (mean_ymax_norm - mean_ymin_norm), 5)

def draw_heatmap(df, class_name, heatmap_size):
    heatmap = np.zeros(shape = heatmap_size)
    heatmap = heatmap.astype(np.float64)
    df_class_name = df[df["class_name"] == class_name]
    mean_bboxes_area_norm = get_mean_bboxes_area_norm(df_class_name, heatmap_size)
    num_bboxes = df_class_name.shape[0]
    for _, row in df_class_name.iterrows():
        if num_bboxes <= 100:
            heatmap[row.get("ymin_norm") : row.get("ymax_norm"), row.get("xmin_norm") : row.get("xmax_norm")] += 2
        elif 100 < num_bboxes <= 255:
            heatmap[row.get("ymin_norm") : row.get("ymax_norm"), row.get("xmin_norm") : row.get("xmax_norm")] += 1
        elif 255 < num_bboxes <= 1000:
            heatmap[row.get("ymin_norm") : row.get("ymax_norm"), row.get("xmin_norm") : row.get("xmax_norm")] = heatmap[row.get("ymin_norm") : row.get("ymax_norm"), row.get("xmin_norm") : row.get("xmax_norm")] + 0.5
        else:
            heatmap[row.get("ymin_norm") : row.get("ymax_norm"), row.get("xmin_norm") : row.get("xmax_norm")] = heatmap[row.get("ymin_norm") : row.get("ymax_norm"), row.get("xmin_norm") : row.get("xmax_norm")] + 0.25           
    return heatmap, num_bboxes, mean_bboxes_area_norm

def nms_wbf(df, img_id, params, technique, iou_thr, skip_box_thr):
    img = cv2.imread(os.path.join(params["IMG_DIR"], img_id))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    img_annotations = df[df["image_file"] == img_id]
    classes_name_full = img_annotations["class_name"].tolist()
    rad_ids = img_annotations["rad_id"].tolist()
    classes_id_full = [params["CLASSES_NAME"].index(i) for i in classes_name_full]
    bboxes_full = img_annotations[["x_min", "y_min", "x_max", "y_max"]].to_numpy().tolist()
    classes_id_count = Counter(classes_id_full)
    classes_id_unique = [params["CLASSES_NAME"].index(i) for i in img_annotations["class_name"].unique().tolist()] 
    norm_scale = np.hstack([width, height, width, height])

    # prepare input data of nms function
    bboxes_lst = []
    scores_lst = []
    classes_id_lst = []
    weights = [] # all weigths is 1, cause all of bounding boxes is labeled by doctor

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
        if technique == "NMS":
            boxes_nms, scores, classes_ids_nms = nms(boxes = bboxes_lst, scores = scores_lst, labels = classes_id_lst,
                                                    iou_thr = iou_thr, weights = weights)
        elif technique == "WBF":
            boxes_nms, scores, classes_ids_nms = weighted_boxes_fusion(boxes_list = bboxes_lst, scores_list = scores_lst, labels_list = classes_id_lst,
                                                                       iou_thr = iou_thr, weights = weights, skip_box_thr = skip_box_thr)

        boxes_nms = boxes_nms * norm_scale
        boxes_nms = boxes_nms.astype(int).tolist()
        classes_ids_nms = classes_ids_nms.astype(int).tolist()

    # add with single class
    boxes_nms.extend(bboxes_single_lst)
    classes_ids_nms.extend(classes_id_single_lst)
    classes_name_nms = [params["CLASSES_NAME"][j] for j in classes_ids_nms]
    img_after = img.copy()

    # draw rad_ids
    img = draw_bbox_rad(img, bboxes_full, classes_name_full, params["CLASSES_NAME"], rad_ids, 
                        params["COLOR"], params["THICKNESS"])
    img_after = draw_bbox(img_after, boxes_nms, classes_name_nms, params["CLASSES_NAME"], 
                        params["COLOR"], params["THICKNESS"])
    meta_info = {"number_bboxes" : len(classes_name_full),
                 "number_bboxes_after" : len(classes_name_nms), 
                 "number_unique_classes" : len(np.unique(np.array(classes_name_full))),
                 "number_unique_classes_after" : len(np.unique(np.array(classes_name_nms)))}
    return img, img_after, meta_info

def compute_iou(bbox_1, bbox_2):
    '''
    Args:
        + bbox_1: (list) with shape (4, ) contain (x_min, y_min, x_max, y_max)
        + bbox_2: (list) with shape (4, ) contain (x_min, y_min, x_max, y_max)

    Output:
        + iou (float) Intersection over Union of both bboexs.
    '''
    
    assert bbox_1[0] < bbox_1[2], "invalid format"
    assert bbox_1[1] < bbox_1[3], "invalid format"
    assert bbox_2[0] < bbox_2[2], "invalid format"
    assert bbox_2[1] < bbox_2[3], "invalid format"

    # determine the coordinates of the intersection rectangle
    x_left = max(bbox_1[0], bbox_2[0])
    x_right = min(bbox_1[2], bbox_2[2])
    y_top = max(bbox_1[1], bbox_2[1])
    y_bottom = min(bbox_1[3], bbox_2[3])
    if (x_left >= x_right) or (y_top >= y_bottom):
        iou = -1
    else:
        # compute intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute area of 2 bboxes
        bbox_1_area = (bbox_1[2] - bbox_1[0]) * (bbox_1[3] - bbox_1[1])
        bbox_2_area = (bbox_2[2] - bbox_2[0]) * (bbox_2[3] - bbox_2[1])
        try:
            # compute iou
            # might be prevent some case iou < 0 or iou > 1 --> but in chest_xray we don't need
            iou = intersection_area / float(bbox_1_area + bbox_2_area - intersection_area)
        except ZeroDivisionError:
            iou = -1
    return iou