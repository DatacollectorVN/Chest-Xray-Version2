import numpy as np
import cv2
import matplotlib.pyplot as plt
import streamlit as st
import os
import pandas as pd
from tqdm import tqdm 
from ensemble_boxes import nms, weighted_boxes_fusion
from collections import Counter
from detectron2.structures import BoxMode
import random
import sys

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

def filter_img_ids(df, img_ids):
    for i, img_id in tqdm(enumerate(img_ids), total = len(img_ids)):
        if i == 0:
            df_final = df[df["image_file"] == img_id]
        else:
            df_final = pd.concat([df_final, df[df["image_file"] == img_id]])
    return df_final

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

def get_chestxray_dicts(df, class_name, img_dir):
    COCO_detectron2_list = [] # list(dict())
    img_ids = df["image_file"].unique().tolist()
    for i, img_id in tqdm(enumerate(img_ids), total = len(img_dir)):
        img_path = os.path.join(img_dir, img_id)        
        assert len(df[df["image_file"] == img_id]["image_width"].unique()) == 1, "Wrong dataset"
        assert len(df[df["image_file"] == img_id]["image_height"].unique()) == 1, "Wrong dataset"
        width = df[df["image_file"] == img_id]["image_width"].unique()[0]
        height = df[df["image_file"] == img_id]["image_height"].unique()[0]
        id_ = i + 1
        img_classes_name = df[df["image_file"] == img_id]["class_name"].values.tolist()
        img_bboxes = df[df["image_file"] == img_id][["x_min", "y_min", "x_max", "y_max"]].values
        x_min = img_bboxes[:, 0]
        y_min = img_bboxes[:, 1]
        x_max = img_bboxes[:, 2]
        y_max = img_bboxes[:, 3]
        annotaions = [] # list(dict())
        for j, img_class_name in enumerate(img_classes_name):
            annotaions_dct = {"bbox" : [x_min[j], y_min[j], x_max[j], y_max[j]],
                              "bbox_mode" : BoxMode.XYXY_ABS,
                              "category_id" : class_name.index(img_class_name)
                             }
            annotaions.append(annotaions_dct)
        COCO_detectron2_dct = {"image_id" : id_,
                               "file_name" : img_path,
                               "height" : height,
                               "width" : width,
                               "annotations" : annotaions
                              }
        COCO_detectron2_list.append(COCO_detectron2_dct)
    return COCO_detectron2_list

def plot_multi_imgs(imgs, # 1 batchs contain multiple images
                    cols = 2, size = 10, # size of figure
                    is_rgb = True, title = None, cmap = "gray",
                    img_size = None): # set img_size if you want (width, height)
    rows = (len(imgs) // cols) + 1
    fig = plt.figure(figsize = (size *  cols, size * rows))
    for i , img in enumerate(imgs):
        if img_size is not None:
            img = cv2.resize(img, img_size)
        fig.add_subplot(rows, cols, i + 1) # add subplot int the the figure
        plt.imshow(img, cmap = cmap) # plot individual image
    plt.suptitle(title)

def convert_bboxes_xyxy_to_xywh_COCO(bboxes_xyxy):
    bboxes_xyxy = np.array(bboxes_xyxy)
    x = np.expand_dims(bboxes_xyxy[:, 0], 1)
    y = np.expand_dims(bboxes_xyxy[:, 1], 1)
    w = np.expand_dims(bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0], 1)
    h = np.expand_dims(bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1], 1)
    bboxes_xywh = np.hstack((x, y, w, h)).astype(np.float32).tolist()
    return bboxes_xywh

def padding(img, bboxes):
    h, w = img.shape[:2]
    if h < w: # padding according to height
        delta = w - h
        scale_bbox = np.array([0, int(delta/2), 0, int(delta/2)])
        bboxes = (np.array(bboxes) + scale_bbox).tolist()
        canvas = np.zeros(shape = (w, w, 3), dtype = np.uint8)
        canvas[int(delta/2):int(delta/2)+h,:, :] = img
    elif w < h: # padding according to width
        delta = h- w
        scale_bbox = np.array([int(delta/2), 0, int(delta/2), 0])
        bboxes = (np.array(bboxes) + scale_bbox).tolist()
        canvas = np.zeros(shape = (h, h, 3), dtype = np.uint8)
        canvas[:,int(delta/2):int(delta/2)+w, :] = img
    elif w == h:
        return img, bboxes
    return canvas, bboxes

### AUGMENTATION
# src: https://blog.paperspace.com/data-augmentation-for-object-detection-rotation-and-shearing/
def draw_bboxes(img, bbox, label, color, thickness):
    ''' draw the rectangle bboxes and its label
    Args:
        img: (np.array) with RGB and shape (H, W, C)
        bbox:(np.array) individual bbox with shape (4, ) with 4 represent to x_min, y_min, x_max. y_max
        label: (str): name of bbox
        color: (iterable) contain the color of its bbox 
        thickness: (int)
    '''
    cv2.rectangle(img, pt1 = (int(bbox[0]), int(bbox[1])), 
                  pt2 = (int(bbox[2]), int(bbox[3])), color = color, thickness = thickness)
    cv2.putText(img, text = label, org = (int(bbox[0]), int(bbox[1] - 5)), 
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.6, color = color, 
                        thickness = 1, lineType = cv2.LINE_AA)

def bboxes_area(bboxes):
    '''Calculate the bounding boxes area
    Args:
        bboxes: (ndarray): contain bounding boxes with shape (N, 4) with
                N is number of boxes and 4 is represent to x_min, y_min, x_max, y_max.
    outputs: 
        area: (ndarray): contain the area of bounding boxes with shape (N, ) 
    '''
    area_bboxes = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    return area_bboxes
    
def clip_bboxes(bboxes, clip_box, labels, alpha):
    """ Clip the bounding boxes to the border of an image 
    Args:
        bboxes: (ndarray) Array contain bounding boxes with shape (N, 4)
                with N is number of bounding boxes and 4 represent to x_min, y_min, x_max. y_max.
        clip_box: (iterable) if array, it have shape (4, ) specifying the diagonal co-ordinates of the image.
                  The coordinates are represented in the formate x_min, y_min, x_max, y_max.
                  Almose case, clip_box = [0, 0, img.shape[1] (img_width), img.shape[0] (img_height)]
        labels: (ndarray) Array contatin the name of bboxes corresponding to bboxes.
        alpha: (float) The configuration. If the percentage loss area of bbox after transform is smaller 
               than alpha then drop this bbox. Otherwise, remain bbox.
    
    Output:
        bboxes: (ndarray) Array containing **clipped** bounding boxes of shape `N X 4` where N is the 
                number of bounding boxes left are being clipped and the bounding boxes are represented in the
                format x_min, y_min, x_max, y_max.
        labels (ndarray)
    """

    bboxes = bboxes.copy()
    area_bboxes = bboxes_area(bboxes)
    
    # convert new x_min, y_min, x_max, y_max inside the border of an image after scale
    x_min = np.maximum(bboxes[:,0], clip_box[0]).reshape(-1, 1)
    y_min = np.maximum(bboxes[:,1], clip_box[1]).reshape(-1, 1)
    x_max = np.minimum(bboxes[:,2], clip_box[2]).reshape(-1, 1)
    y_max = np.minimum(bboxes[:,3], clip_box[3]).reshape(-1, 1)

    bboxes = np.hstack((x_min, y_min, x_max, y_max))

    # compute the percentage of bboxes area after
    #print(f'bboxes_area(bboxes) = {bboxes_area(bboxes)}')
    #print(f'area_bboxes = {area_bboxes}')
    percen_area_bboxes = bboxes_area(bboxes) / area_bboxes

    # the percentage of loss area
    percen_loss_area_bboxes = 1 - percen_area_bboxes
    #print(f'loss_area_bboxes = {percen_loss_area_bboxes}')
    mask = (percen_loss_area_bboxes < alpha).astype(int)
    #print(f'mask = {mask}')
    
    # remain the boxes with satisfied condition with corespoding to index
    bboxes = bboxes[mask == 1, :]
    labels = labels[mask == 1]
    

    return bboxes, labels

def rotate_img(img, angle):
    '''Rotate the image
    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black.

    Args:
        img: (nd.array) image with shape (H, W, C)
        angle: (float) angle by which the image is to be rotated. 
    
    Output:
        img: (nd.array) Rotated image.
    '''

    # get the coordinates of center point of image
    h, w = img.shape[0], img.shape[1]
    center_x, center_y = w // 2, h // 2

    # transform matrix (applying the negative of the
    # angle to rotate clockwise)
    rotation_matrix = cv2.getRotationMatrix2D(center = (center_x, center_y), 
                                              angle = angle, scale = 1.0)
    
    # get the sine and cosine (cosine = alpha and sine = beta cause scale = 1.0)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])

    # calculate the new bounding boxes dimentions of the image. 
    # (explain in document)
    new_w = int((w * cos) + (h * sin))
    new_h = int((w * sin) + (h + cos))

    # calcuate the new center point with new width and height
    new_center_x, new_center_y = new_w // 2, new_h // 2

    '''For ensuring the the center of the new image does not move since it is the axis of rotation itself'''
    delta_center_x, delta_center_y = (new_center_x - center_x, new_center_y - center_y)

    # adjust the rotation matrix tot ake into account translation
    rotation_matrix[0, 2] += delta_center_x
    rotation_matrix[1, 2] += delta_center_y

    # perform the actual rotation and return the image
    img = cv2.warpAffine(img, rotation_matrix, (new_w, new_h))

    return img 

def convert_2_to_4_corners(bboxes):
    """Get 4 corrners of bounding boxes
    
    Args:
        bboxes: (ndarray) Array contain bounding boxes with shape (N, 4)
                with N is number of bounding boxes and 4 represent to x_min, y_min, x_max. y_max.
        
    Ouput: 
        corners: (ndarray) Array contatin bounding boxes with shape (N, 8)
                with N is number of bounding boxes and 8 represent to x1, y1, x2, y2, x3, y3, x4, y4
    """

    # get width and height of bboxes 
    width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)

    # top-left corner
    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)

    # top-right corner
    x2 = x1 + width
    y2 = y1

    # bottom-left corner
    x3 = x1
    y3 = y1 + height

    # bottom-right corner
    x4 = bboxes[:, 2].reshape(-1, 1)
    y4 = bboxes[:, 3].reshape(-1, 1)

    # stack all of these
    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners

def convert_4_to_2_corners(corners_bboxes):
    """ Get 2 corrners of bounding boxes
    
    Args:
        corners_bboxes: (ndarray) Numpy array of shape `N x 8` containing N bounding boxes each described by their 
                                          corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  
    
    Output:
        bboxes: (ndarray)  Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the 
                                    number of bounding boxes and the bounding boxes are represented in the
                                    format x_min, y_min, x_max, y_max
    """
    x_axis = corners_bboxes[:, [0, 2, 4, 6]]
    y_axis = corners_bboxes[:, [1, 3, 5, 7]]

    # get x_min, y_min, x_max, y_max
    x_min = np.min(x_axis, axis = 1).reshape(-1, 1)
    y_min = np.min(y_axis, axis = 1).reshape(-1, 1)
    x_max = np.max(x_axis, axis = 1).reshape(-1, 1)
    y_max = np.max(y_axis, axis = 1).reshape(-1, 1)

    # convert it to 2 corners
    bboxes = np.hstack(tup = (x_min, y_min, x_max, y_max))

    return bboxes
    
def rotate_bboxes(corners_bboxes, angle, center_x, center_y, w, h):
    """Rotate the bounding boxes.
    
    Args: 
        corners_bboxes: (ndarray) Numpy array of shape (N , 8) containing N bounding boxes each described by their 
                        corner coordinates `x1 y1 x2 y2 x3 y3 x4 y4`
        angle: (float) Angle by which the image is to be rotated
        center_x: (int) x coordinate of the center of image (about which the box will be rotated)
        center_y: (int) y coordinate of the center of image (about which the box will be rotated)
        w: (int) Width of the image
        h: (int) Height of the image
    
    Output:
        rotated_corners_bboxes: (ndarray) Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
                                corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """

    '''Exaplain detail this function in document'''
    # reshape the corners_bboxes 
    corners_bboxes = corners_bboxes.reshape(-1, 2)
    corners_bboxes = np.hstack((corners_bboxes, np.ones(shape = (corners_bboxes.shape[0], 1), dtype = np.float32)))

    # transform matrix (applying the negative of the
    # angle to rotate clockwise)
    rotation_matrix = cv2.getRotationMatrix2D(center = (center_x, center_y), 
                                              angle = angle, scale = 1.0)

    # get the sine and cosine (cosine = alpha and sine = beta cause scale = 1.0)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])

    # calculate the new bounding boxes dimentions of the image. 
    # (explain in document)
    new_w = int((w * cos) + (h * sin))
    new_h = int((w * sin) + (h + cos))

    # calcuate the new center point with new width and height
    new_center_x, new_center_y = new_w // 2, new_h // 2

    '''For ensuring the the center of the new image does not move since it is the axis of rotation itself'''
    delta_center_x, delta_center_y = (new_center_x - center_x, new_center_y - center_y)

    # adjust the rotation matrix tot ake into account translation
    rotation_matrix[0, 2] += delta_center_x
    rotation_matrix[1, 2] += delta_center_y

    # corners_bboxes after transform (dot product)
    rotated_corners_bboxes = np.dot(rotation_matrix, corners_bboxes.T).T

    # reshape it to the original shape (N, 8)
    rotated_corners_bboxes = rotated_corners_bboxes.reshape(-1, 8)

    return rotated_corners_bboxes

def letterbox_img(img, inp_dim):
    """Resize image with unchanged aspect ratio using padding
    
    Args:
        img: (ndarray) Original image RBG with shape (H, W, C)
        inp_dim: (ndarray) with (width, height) 
    
    Output:
        canvas: (ndarray) Array with desired size and contain the image at the center
    """
    
    # get width and height of original image
    img_w, img_h = img.shape[1], img.shape[0]
    
    # get width and height of desierd dimension
    w, h = inp_dim 

    # calculate the new_w and new_h of content of image inside the desired dimension
    # calculate the scale_factor according to width or heigh --> get the minimum value
    # cause we want to maintain the information of orginal image.
    scale_factor = min(w / img_w, h / img_h)
    new_w = int(img_w * scale_factor)
    new_h = int(img_h * scale_factor)

    # resized orginal image
    # why cv2.INTER_CUBIC ?
    # https://stackoverflow.com/questions/23853632/which-kind-of-interpolation-best-for-resizing-image
    # https://chadrick-kwag.net/cv2-resize-interpolation-methods/
    resized_img = cv2.resize(src = img, dsize = (new_w, new_h), 
                             interpolation = cv2.INTER_CUBIC)
    
    # create the canvas
    canvas = np.full(shape = (inp_dim[1], inp_dim[0], 3), fill_value = 128)
    
    # paste the image on the canvas (at center)
    # canvas[top : bottom, left : right, :]
    top = (h - new_h) // 2
    bottom = top + new_h
    left = (w - new_w) // 2
    right = left + new_w
    canvas[top : bottom, left : right, :] = resized_img

    return canvas

class Rotate(object):
    ''' Rotates an images
    Bounding boxes which habe an area og less than 25% in the remaining in the 
    transformed image is dropped (removed). The resolution is maintained, and the remaining
    area if nay is filled by black color.

    Args:
        angle: (float) Angle by which the image is to be rotated
        p: (float) The probability with which the image is rotated.

    Output: 
        img: (ndarray) The scaled image with shape (H, W, C)
        bboxes: (ndarray) The transformed bounding boxe with shape (N, 4)
                N is number of bounding boxes and 4 represents x_min, y_min, x_max, y_max.
        labels: (ndarray) the transformed labels
    '''

    def __init__(self, angle=20, p=0.5):
        self.angle = angle
        self.p = p
    
    def __call__(self, img, bboxes, labels):
        img = img.copy()
        bboxes = bboxes.copy()
        labels = labels.copy()

        if random.random() < self.p:
            # get random angle from range
            angle = self.angle
            
            # get the width, height and center of point of image
            h, w = img.shape[0], img.shape[1]
            center_x, center_y = w // 2, h // 2

            # rotate the image
            img = rotate_img(img, angle)

            # convert 2 corners of boundig boxes to 4 corners (easy to calculate)
            corners = convert_2_to_4_corners(bboxes)
            
            # rotate bounding boxes
            corners = rotate_bboxes(corners_bboxes = corners, angle = angle, 
                                    center_x = center_x, center_y = center_y, 
                                    w = w, h = h)
            new_bboxes = convert_4_to_2_corners(corners)

            # calculate the scale_factor after rotation image (cause its dimension is change after rotation)
            scale_factor_x = img.shape[1] / w
            scale_factor_y = img.shape[0] / h

            # but we don't want to change the dimension of final image --> resize it to original dimension
            img = cv2.resize(img, (w, h))

            # then we must convert the coordinate of new_bboxes after rotation
            new_bboxes[:, :4] /= [scale_factor_x, scale_factor_y,scale_factor_x, scale_factor_y]
            bboxes = new_bboxes

            bboxes, labels = clip_bboxes(bboxes, clip_box = [0, 0, 1 + w, 1 + h], labels = labels,
                                                alpha = 0.25)
            
        return img, bboxes, labels


class RandomHorizontalFlip(object):
    '''Radnomly horizontally flip the Image with the probability p.
    Args:
        p: (float) The probability with which the image is flipped.

    Output: 
        img: (np.array) The flipped image with shape (HxWxC)
        bboxes: (np.array) The transformed bounding boxe with shape (N, 4)
                N is number of bounding boxes and 4 represents x_min, y_min, x_max, y_max.
        labels: (ndarray) the transformed labels
    '''
    
    def __init__(self, p):
        self.p = p

    def __call__(self, img, bboxes, labels):
        '''get the img_center coordinate''' 
        # img with shape (512, 256, 3) (H = 512, W = 256)
        # img_center = [128., 256.] 
        # [::-1] cause the img_center[0] is x_center and img_center[1] is y_center.
        img_center = np.array(img.shape[:2])[::-1] / 2
        #horizontal stack --> return standard format in image center
        # (x_center, y_center, x_center, y_center) like (x_min, y_min, x_max, y_max)
        img_center = np.hstack((img_center, img_center))
        bboxes = bboxes.copy()
        img = img.copy()
        if random.random() < self.p:
            # filp the img
            img = img[:, ::-1, :] # horizontal flip --> flip the width
            # https://stackoverflow.com/questions/59861233/cv2-rectangle-calls-overloaded-method-although-i-give-other-parameter
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # if dont --> make error when draw by opencv :>
            
            # flip the bboxes (Explain in document)
            # x_min, x_max change
            bboxes[:, [0, 2]] += 2 * (img_center[[0, 2]] - bboxes[:, [0, 2]])
            bbox_w = abs(bboxes[:, 0] - bboxes[:, 2])
            # convert the x_min in the top left and x_max in the bottom right.
            # x_min
            bboxes[:, 0] -= bbox_w
            # x_max
            bboxes[:, 2] += bbox_w
        
        return img, bboxes, labels


class Shear(object):
    """ Randomly shears an image in the horizontal direction
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Args:
        shear_factor: (float or tuple(float))
                      If **float**, the image is sheared horizontally by a factor drawn 
                      randomly from a range (-`shear_factor`, `shear_factor`). 
                      If **tuple**, the `shear_factor` is drawn randomly from values specified by the 
                      tuple.
        p: (float) The probability with which the image is rotated.
    
    Output:
        img: (ndarray) The scaled image with shape (H, W, C)
        bboxes: (ndarray) The transformed bounding boxes with shape (N, 4)
                N is number of bounding boxes and 4 represents x_min, y_min, x_max, y_max.
        labels: (ndarray) the transformed labels
    """

    def __init__(self, shear_factor=0.2, p=0.5):
        self.shear_factor = shear_factor
        self.p = p
    
    def __call__(self, img, bboxes, labels):
        img = img.copy()
        bboxes = bboxes.copy()
        labels = labels.copy()

        if random.random() < self.p:
            shear_factor = self.shear_factor
            # get the width, height of image
            h, w = img.shape[0], img.shape[1]

            # in document explain why we horizontal flip image when shear is negative
            '''
            if shear_factor < 0:
            1. Horizontal flip image.
            2. Shearing image with positive shear_factor.
            3. Horizontal flip image again
            '''

            if shear_factor < 0:
                img, bboxes, labels = RandomHorizontalFlip(p = 1)(img, bboxes, labels)
            
            # create the shearing matrix
            shear_matrix = np.array([[1, abs(shear_factor), 0], 
                                     [0, 1, 0]])
            
            # when shearing, for maintaining the information of image 
            # then the image's width will change
            new_w = w + abs(shear_factor * h)

            # the cooridnate of bounding boxes (x_min, x_max) change
            bboxes[:, [0, 2]] += ((bboxes[:, [1, 3]]) * abs(shear_factor)).astype(int)

            # shearing the image 
            img = cv2.warpAffine(img, shear_matrix, dsize = (int(new_w), h))

            # if shear_factor < 0, we flip it again
            if shear_factor < 0:
                img, bboxes, labels = RandomHorizontalFlip(p = 1)(img, bboxes, labels)

            # resize the image (after shearing) into original size
            img = cv2.resize(img, (w, h))

            # cause we resize the image --> resize the coordinates of bboxes
            scale_factor_x = new_w / w
            bboxes[:, :4] /= [scale_factor_x, 1, scale_factor_x, 1]

            # clip the bboxes
            bboxes, labels = clip_bboxes(bboxes, clip_box = [0, 0, 1 + w, 1 + h], labels = labels,
                                                alpha = 0.25)
        return img, bboxes, labels