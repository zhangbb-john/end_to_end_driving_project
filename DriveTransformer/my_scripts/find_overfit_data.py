import os
import json

dev_220 = "/home/slxy/zca/code/Bench2Drive/leaderboard/data/bench2drive220.xml"

# full_json = "/home/slxy/Downloads/bench2drive_base_1000.json"
full_json = "/home/slxy/Downloads/bench2drive_full+sup_13638.json"

import xml.etree.ElementTree as ET

xml_path = dev_220  # 改成你的文件路径

tree = ET.parse(xml_path)
root = tree.getroot()

results = []
for route in root.findall("route"):
    route_id = route.get("id")
    town = route.get("town")
    results.append((route_id, town))

path = []
for rid, town in results:
    # print(f"route_id={rid}, town={town}")
    path.append((town, rid))

with open(full_json, 'r') as f:
    data = json.load(f)


cnt = 0
# print(data)

download_set = set()

for item in path:
    # sub_path = f"{item[0]}_Route{item[1]}"
    
    for k in data.keys():
        if item[0] in k.split('_') and f"Route{item[1]}" in k.split('_'):
            sub_path = f"{item[0]}_Route{item[1]}"
            print(sub_path)
            cnt+=1
            download_set.add(k)
print(f"cnt is :{cnt}")

with open('overfit.txt', 'w') as f:
    f.write("\n".join(download_set))