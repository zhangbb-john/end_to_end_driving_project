import os
import json

data_root = "/data/Bench2Drive/bench2drive-base"
data_list = os.listdir(data_root)
# data_list = [item for item in data_list if item.endswith(".tar.gz")]
print(f"len(data_list):{len(data_list)}")

json_file = "/data/Bench2Drive/bench2drive_base_1000.json"
with open(json_file, 'r') as f:
    data_json = json.load(f)
print(f"len(data_json):{len(data_json)}")
# print(data_json.keys())

for item in data_json.keys():
    if item.split('.')[0] not in data_list:
        print(f"miss : {item}")
    
# for item in data_list:
#     if item not in data_json.keys():
#         print(f"miss2:{item}")