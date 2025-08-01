import os
import random

# 数据集路径配置
data_dir = "VOC2007_flood"
image_dir = os.path.join(data_dir, "JPEGImages")
image_sets_dir = os.path.join(data_dir, "ImageSets", "Segmentation")


def generate_voc_split_files(images_dir, output_dir, train_ratio=0.8, random_seed=42):
    """更新VOC格式的划分文件（train.txt, val.txt）"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图像文件名（不含扩展名）
    image_files = [os.path.splitext(f)[0] for f in os.listdir(images_dir)
                   if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("错误: JPEGImages目录为空!")
        return [], []

    # 随机划分
    random.seed(random_seed)
    random.shuffle(image_files)
    train_size = int(len(image_files) * train_ratio)
    train_files = image_files[:train_size]
    val_files = image_files[train_size:]

    # 写入train.txt（覆盖旧文件）
    with open(os.path.join(output_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_files))

    # 写入val.txt（覆盖旧文件）
    with open(os.path.join(output_dir, "val.txt"), "w") as f:
        f.write("\n".join(val_files))

    print(f"已更新划分文件: train={len(train_files)}, val={len(val_files)}")
    print(f"文件位置: {output_dir}")
    return train_files, val_files


if __name__ == "__main__":
    # 仅更新数据集划分文件
    print("开始更新数据集划分文件...")
    train_files, val_files = generate_voc_split_files(image_dir, image_sets_dir)
    print("更新完成!")