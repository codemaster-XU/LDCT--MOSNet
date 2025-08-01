import os
import shutil
import random
from PIL import Image
import numpy as np

# 修正后的路径（使用原始字符串）
input_base = r"D:\Program Files\OneDrive\文档\nnUNet-master\nnUNet-master\lidc数据集"
output_base = r"D:\Program Files\OneDrive\Desktop"

# 根据图片中的实际结构设置路径
image_dir = os.path.join(input_base)  # 原图直接在lidc数据集目录下 (LIDC_*.png)
mask_dir = os.path.join(input_base, "mask")  # mask在mask子目录下 (LIDC_Mask_*.png)

# 创建VOC标准目录结构
voc_dir = os.path.join(output_base, "VOCdevkit", "VOC2007")
dirs_to_create = [
    os.path.join(voc_dir, "JPEGImages"),
    os.path.join(voc_dir, "SegmentationClass"),
    os.path.join(voc_dir, "ImageSets", "Segmentation")
]

for d in dirs_to_create:
    os.makedirs(d, exist_ok=True)

# 获取文件列表 - 基于图片中的实际命名格式
# 原图: LIDC_<数字>.png 直接在主目录
# mask: mask/LIDC_Mask_<数字>.png
image_files = sorted([
    f for f in os.listdir(input_base)
    if f.startswith('LIDC_') and f.endswith('.png') and not f.startswith('LIDC_Mask_')
])

mask_files = sorted([
    f for f in os.listdir(mask_dir)
    if f.startswith('LIDC_Mask_') and f.endswith('.png')
])

print(f"找到 {len(image_files)} 张原图 | 首尾: {image_files[0]}...{image_files[-1] if image_files else ''}")
print(f"找到 {len(mask_files)} 个mask | 首尾: {mask_files[0]}...{mask_files[-1] if mask_files else ''}")

# 按文件名排序后的顺序匹配
file_pairs = []
if len(image_files) == len(mask_files):
    # 文件名数量相等，按排序后的顺序匹配
    sorted_images = sorted(image_files)
    sorted_masks = sorted(mask_files)

    for img, mask in zip(sorted_images, sorted_masks):
        file_pairs.append((img, mask))
        print(f"配对: {img} ↔ {mask}")
elif image_files and mask_files:
    # 文件名数量不等，尝试通过数字ID匹配
    print("警告: 原图与mask数量不一致，尝试数字ID匹配")

    # 提取原图数字ID
    img_ids = {}
    for f in image_files:
        try:
            parts = f.split('_')
            if len(parts) > 1:
                img_id = int(parts[1].split('.')[0])
                img_ids[img_id] = f
        except:
            pass

    # 提取mask数字ID
    mask_ids = {}
    for f in mask_files:
        try:
            parts = f.split('_')
            if len(parts) > 2:
                mask_id = int(parts[2].split('.')[0])
                mask_ids[mask_id] = f
        except:
            pass

    # 匹配相同ID的文件
    common_ids = set(img_ids.keys()) & set(mask_ids.keys())
    for id in sorted(common_ids):
        img_file = img_ids[id]
        mask_file = mask_ids[id]
        file_pairs.append((img_file, mask_file))
        print(f"ID匹配: [{id}] {img_file} ↔ {mask_file}")
else:
    print("错误: 未找到可匹配的文件")

# 处理并复制文件
all_valid_names = []
if file_pairs:
    for idx, (img_file, mask_file) in enumerate(file_pairs):
        # 生成6位数字文件名
        new_name = f"{idx + 1:06d}"

        # 处理原图
        img_src = os.path.join(input_base, img_file)
        img_dest = os.path.join(voc_dir, "JPEGImages", f"{new_name}.jpg")
        try:
            # 转换为JPG格式并保存
            img = Image.open(img_src)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(img_dest)
            all_valid_names.append(new_name)
        except Exception as e:
            print(f"处理图像 {img_file} 时出错: {str(e)}")
            continue

        # 处理mask
        mask_src = os.path.join(mask_dir, mask_file)
        mask_dest = os.path.join(voc_dir, "SegmentationClass", f"{new_name}.png")
        try:
            mask = Image.open(mask_src)
            # 确保mask为单通道索引图
            if mask.mode != 'L':
                mask = mask.convert('L')
            # 将像素值转换为类别索引
            mask_array = np.array(mask)
            unique_vals = np.unique(mask_array)
            # 如果只有两种值(通常是0和255)，转换为二值(0和1)
            if len(unique_vals) == 2 and max(unique_vals) > 1:
                mask_array = (mask_array > 0).astype(np.uint8)

            Image.fromarray(mask_array).save(mask_dest)
            print(f"成功转换: {img_file} → {new_name}.jpg  |  {mask_file} → {new_name}.png")
        except Exception as e:
            print(f"处理mask {mask_file} 时出错: {str(e)}")
            continue

# 数据集划分
if all_valid_names:
    random.shuffle(all_valid_names)
    total = len(all_valid_names)
    train_end = int(total * 0.7)
    val_end = train_end + int(total * 0.15)
    test_end = total - train_end - val_end

    splits = {
        "train": all_valid_names[:train_end],
        "val": all_valid_names[train_end:train_end + val_end],
        "test": all_valid_names[train_end + val_end:] if test_end > 0 else []
    }

    # 保存划分文件
    for split_name, names in splits.items():
        if names:
            with open(os.path.join(voc_dir, "ImageSets", "Segmentation", f"{split_name}.txt"), "w") as f:
                f.write("\n".join(names))

    print(f"\n转换完成！共处理 {len(all_valid_names)} 个样本")
    print(f"数据集已保存至: {voc_dir}")
    print(f"训练集: {len(splits['train'])} 个样本")
    print(f"验证集: {len(splits['val'])} 个样本")
    print(f"测试集: {len(splits['test'])} 个样本")
else:
    print("\n错误：未找到匹配的图像-掩码对")
    print("可能原因：")
    print("1. 文件名格式不符（原图应为LIDC_XXXX.png，mask应为LIDC_Mask_YYYY.png）")
    print("2. 文件数量不一致（原图与mask数量不同）")
    print("3. 文件路径错误")
    print(f"原始目录: {input_base}")
    print(f"mask目录: {mask_dir}")