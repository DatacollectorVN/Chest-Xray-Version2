# Chest-Xray-Version2
After many experiments, we used RetinaNet and FasterR-CNN with FPN backbone, we concluded the RetinaNet is better.

**Best parameters of backbone:**
+ MODEL.FPN.OUT_CHANNELS: 256
+ MODEL.ROI_HEADS.IN_FEATURES: ['res4']

## 14 Classes:
RetinaNet with ResNet50 in FPN backbone (3x) give us the best results in validation set. 

**Parameters of config:**
+ Augmentations: ResizeShortestEdge(short_edge_length=(640, 672, 704, 736, 768, 800),max_size=1333, sample_style='choice')
+ IMS_PER_BATCH: 1
+ BATCH_SIZE_PER_IMAGE: 64
+ WARMUP_ITERS: 100
+ BASE_LR: 0.001
+ MAX_ITER: 2000
+ STEPS_MIN: 200
+ STEPS_MAX: 1900
+ GAMMA: 0.1
+ LR_SCHEDULER_NAME: WarmupCosineLR

**Results:**
+ mAP@[0.5:0.95]: 0.11
+ mAP@0.5: 0.21 