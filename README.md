# Chest-Xray-Version2
***Status: Ongoing***

The model detect the abnormalities in chest-Xray image by [Detectron2](https://github.com/facebookresearch/detectron2) - Pytorch.

## INSTALLATION 
1. Create virtual environment.
```bash
conda create -n chestxrayv2 python=3.7
conda activate chestxrayv2
```
2. clone this repository.
3. Install required packages. 
```bash 
pip install -r requirements.txt
```
4. Setup Detectron2.

See [installation instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html). Or see [my instructions](https://github.com/DatacollectorVN/Detectron2-Tutorial).

## DATA PREPROCESSING
See [the document](https://docs.google.com/presentation/d/1oXhtmHP9GB1MmArHH-gxWWOc9mIvz-0N6M3GBCcqx0U/edit?usp=sharing) for understanding how we process the chest-Xray dataset from [VinBigdata](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/overview).

5. Download the standard and additional data after processing.
`bash download_data_standard_add.sh`

You can run file streamlit_.py for exploring the dataset in eda/ or nms-wbf-visualize/ .
  
*Note: Need to configure config/streamlit_eda.yaml file.*

## FOR TRAINING
```bash 
python train.py
```
You can configure traininig in config/train.yaml.

## FOR EVALUATING
```bash
python eval.py
```
You can configure evaluating in config/inference.yaml.
