from pycocotools.coco import COCO
json_path = "/home/sjtu/scratch/tongzhao/e2ec/data/coco/annotations/instances_train2017.json"
img_path = "/home/sjtu/scratch/tongzhao/e2ec/data/coco/train2017"

coco = COCO(annotation_file=json_path)
# 获取对应图像id的所有annotations idx信息
img_id = 295695
ann_ids = coco.getAnnIds(imgIds=img_id)
# ann_ids = 7880
# 根据annotations idx信息获取所有标注信息
targets = coco.loadAnns(ann_ids)

print(targets)