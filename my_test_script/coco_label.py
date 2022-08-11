import json

json_path = "/home/sjtu/scratch/tongzhao/e2ec/data/coco/annotations/instances_val2017.json"
json_labels = json.load(open(json_path, "r"))
print(json_labels["info"])