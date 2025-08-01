import os
import random

# 设置数据集路径
root_path = "autodl-tmp/data1/lidc/VOCdevkit/VOC2007_flood"
image_folder = os.path.join(root_path, "JPEGImages")
train_txt_path = os.path.join(root_path, "ImageSets", "Segmentation", "train.txt")
val_txt_path = os.path.join(root_path, "ImageSets", "Segmentation", "val.txt")

# 获取所有图像文件的名称
image_files = [f.split('.')[0] for f in os.listdir(image_folder) if f.endswith('.jpg')]

# 打乱数据顺序
random.shuffle(image_files)

# 70% 用于训练，30% 用于验证
split_idx = int(len(image_files) * 0.7)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

# 保存 train.txt 和 val.txt 文件
with open(train_txt_path, 'w') as f:
    for image in train_files:
        f.write(image + '\n')

with open(val_txt_path, 'w') as f:
    for image in val_files:
        f.write(image + '\n')

print(f"train.txt 和 val.txt 文件已生成在 {root_path}/ImageSets/Segmentation 目录下")
