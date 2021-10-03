#!/bin/bash
python ./src/download_dataset_chest_xray_standard_add.py
unzip CHEST_XRAY.zip
rm CHEST_XRAY.zip
