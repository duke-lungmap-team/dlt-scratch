import sys
import os
import re
import json
import numpy as np
from PIL import Image

try:
    base_dir = sys.argv[1]
except IndexError:
    sys.exit("Please specify a directory of NumPy sub-regions")

os.chdir(base_dir)

region_files = os.listdir(base_dir)

region_dict = {}

for f in sorted(region_files):
    match = re.search('^(.+)_<(\d+),(\d+)>.(npy|tif)', f)
    base_img_name, x, y, ext = match.groups()

    if ext == 'npy':
        region = np.load(f)
        h, w, d = region.shape
    elif ext == 'tif':
        region = Image.open(f)
        h = region.height
        w = region.width
    else:
        continue

    orig_img_name = ".".join([base_img_name, 'tif'])

    if orig_img_name not in region_dict:
        region_dict[orig_img_name] = []

    rect = {
        'x1': float(x),
        'y1': float(y),
        'x2': float(x) + w,
        'y2': float(y) + h
    }

    region_dict[orig_img_name].append(rect)

# now convert to TensorBox style JSON
tensor_box_json = []

for img, rects in region_dict.items():
    tensor_box_json.append(
        {
            "image_path": img,
            "rects": rects
        }
    )

full_path, parent_dir = os.path.split(os.getcwd())

json_output = ".".join([parent_dir, 'json'])

os.chdir(full_path)

with open(json_output, 'w') as f:
    json.dump(tensor_box_json, f, indent=4)
