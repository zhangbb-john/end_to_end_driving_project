import os
import json

root_path = "/data/Bench2Drive/bench2drive-base"
dest_root = "/home/slxy/zca/temp"

_ANNO = "anno"
_CAMERA = "camera"

# for root, dirs, files in os.walk(root_path):
#     if _ANNO in dirs and _CAMERA in dirs:
#         print(f"root:{root}")
#         case_name = root.split('/')[-1]
#         dest_path = os.path.join(dest_root, case_name)
#         os.system(f"cp {root}/anno/* {dest_root}")

json_list = os.listdir(dest_root)
json_list = [item for item in json_list if not item.endswith('.gz')]

import collections
from tqdm import tqdm
count_dict = collections.defaultdict(int)

for json_name in tqdm(json_list):
    json_path = os.path.join(dest_root, json_name)
    with open(json_path, 'r') as f:
        data_dict = json.load(f)
        print(data_dict)
        for item in data_dict["bounding_boxes"]:
            obj_cls = item['class']
            count_dict[obj_cls] += 1

print(count_dict)
with open('count.json', 'w') as f:
    json.dump(count_dict, f)