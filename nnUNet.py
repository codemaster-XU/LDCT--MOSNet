import os
import numpy as np
import imageio.v2 as imageio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import time
import json
import matplotlib.pyplot as plt
import glob
import torch.nn.functional as F
from tqdm import tqdm
import gc
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_dilation

# ============== 用户配置区域 ==============
# 修改为三个路径对
original_image_dirs = [
    "libc/image"
]

original_mask_dirs = [
    "libc/mask"
]

pixel_spacing = 1.0  # 单位：mm/pixel（可根据实际数据调整）
# ========================================

# 创建输出目录
output_dir = "nnUNet(org.)_output"
os.makedirs(output_dir, exist_ok=True)


# 定义轻量级U-Net模型
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # 编码器
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # 解码器
        self.up3 = self.upconv_block(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.up2 = self.upconv_block(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = self.upconv_block(128, 64)
        self.dec1 = self.conv_block(128, 64)

        # 输出层
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码路径
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # 解码路径
        dec3 = self.up3(enc4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        return torch.sigmoid(self.final(dec1))


# 自定义数据集类
class MedicalDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取图像 - 使用imageio.v2避免警告
        image = imageio.imread(self.image_paths[idx])
        mask = imageio.imread(self.mask_paths[idx])

        # 确保图像是单通道的（CT图像）
        if len(image.shape) > 2:
            # 转换为灰度
            image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        if len(mask.shape) > 2:
            # 转换为灰度
            mask = np.dot(mask[..., :3], [0.2989, 0.5870, 0.1140])

        # 扩展维度，添加通道维度
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # 转换为PyTorch张量并归一化
        image = torch.from_numpy(image).float() / 255.0
        mask = torch.from_numpy(mask).float()

        # 确保mask是二值图（0或1）
        mask = (mask > 0.5).float()

        return image, mask


# 计算基础指标
def calculate_metrics(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    target_bin = target.float()  # 目标已经是二值图

    # 计算混淆矩阵
    tp = torch.sum(pred_bin * target_bin)
    fp = torch.sum(pred_bin * (1 - target_bin))
    fn = torch.sum((1 - pred_bin) * target_bin)
    tn = torch.sum((1 - pred_bin) * (1 - target_bin))

    # 避免除零错误
    epsilon = 1e-7

    # 计算指标
    iou = tp / (tp + fp + fn + epsilon)
    dice = (2 * tp) / (2 * tp + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)

    return {
        "iou": iou.item(),
        "dice": dice.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "accuracy": accuracy.item()
    }


# 计算表面距离指标（优化速度）
def calculate_surface_metrics(pred, target, pixel_spacing):
    # 增加边界检查
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()

    # 检查是否为全黑图像
    if np.sum(pred_np) == 0 or np.sum(target_np) == 0:
        return {"msd": 1000.0, "hd": 1000.0, "assd": 1000.0}

    # 使用距离变换优化计算
    def compute_surface_distances(pred, target):
        # 计算距离变换
        dist_pred = distance_transform_edt(~pred)
        dist_target = distance_transform_edt(~target)

        # 计算从预测边界到目标边界的距离
        surface_to_target = dist_pred * target
        surface_to_pred = dist_target * pred

        # 提取非零像素
        surface_pixels = pred.astype(bool)
        target_pixels = target.astype(bool)

        msd = np.mean(dist_pred[target_pixels]) if np.any(target_pixels) else 1000.0
        asd_pred = np.mean(dist_pred[target_pixels]) if np.any(target_pixels) else 1000.0
        asd_target = np.mean(dist_target[surface_pixels]) if np.any(surface_pixels) else 1000.0
        assd = (asd_pred + asd_target) / 2

        # 计算豪斯多夫距离（采样点优化）
        sample_size = min(100, np.sum(surface_pixels), np.sum(target_pixels))
        pred_points = np.argwhere(surface_pixels)
        target_points = np.argwhere(target_pixels)

        if sample_size > 0:
            # 随机采样以减少计算量
            pred_points = pred_points[np.random.choice(len(pred_points), sample_size, replace=False)]
            target_points = target_points[np.random.choice(len(target_points), sample_size, replace=False)]

            # 计算双向豪斯多夫距离
            hd1 = directed_hausdorff(pred_points, target_points)[0]
            hd2 = directed_hausdorff(target_points, pred_points)[0]
            hd = max(hd1, hd2)
        else:
            hd = 1000.0

        return msd, hd, assd

    # 转换为0-1像素值
    pred_bin = pred_np > 0.5
    target_bin = target_np > 0.5

    # 计算指标
    msd, hd, assd = compute_surface_distances(pred_bin, target_bin)

    return {
        "msd": msd * pixel_spacing,
        "hd": hd * pixel_spacing,
        "assd": assd * pixel_spacing
    }


# 训练函数（单次训练）
def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=100):
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "iou": [],
        "dice": [],
        "precision": [],
        "recall": [],
        "accuracy": [],
        "msd": [],
        "hd": [],
        "assd": []
    }

    best_iou = 0.0
    best_metrics = None

    # 释放GPU内存
    torch.cuda.empty_cache()
    gc.collect()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        num_train_batches = 0

        # 训练阶段
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False)
        for images, masks in train_iter:
            images, masks = images.to(device), masks.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            num_train_batches += 1
            train_iter.set_postfix(loss=loss.item())

        # 计算平均训练损失
        train_loss = train_loss / len(train_loader.dataset)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            val_iter = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]", leave=False)
            for images, masks in val_iter:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

                # 保存用于距离计算
                all_outputs.append(outputs.cpu())
                all_targets.append(masks.cpu())

        # 计算平均验证损失
        val_loss = val_loss / len(val_loader.dataset)

        # 合并所有验证输出和目标
        all_outputs_tensor = torch.cat(all_outputs)
        all_targets_tensor = torch.cat(all_targets)

        # 计算指标（只计算基础指标，距离指标每5个epoch计算一次）
        metrics = calculate_metrics(all_outputs_tensor, all_targets_tensor)
        surface_metrics = {"msd": history["msd"][-1] if history["msd"] else 1000.0,
                           "hd": history["hd"][-1] if history["hd"] else 1000.0,
                           "assd": history["assd"][-1] if history["assd"] else 1000.0}

        # 每5个epoch计算一次距离指标（或第一个/最后一个epoch）
        if epoch % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            # 采样部分数据计算距离指标（减少计算量）
            sample_indices = np.random.choice(len(all_outputs_tensor), min(10, len(all_outputs_tensor)), replace=False)
            sample_outputs = all_outputs_tensor[sample_indices]
            sample_targets = all_targets_tensor[sample_indices]

            msd_sum, hd_sum, assd_sum = 0.0, 0.0, 0.0
            count = 0

            for i in range(len(sample_outputs)):
                sm = calculate_surface_metrics(sample_outputs[i:i + 1], sample_targets[i:i + 1], pixel_spacing)
                msd_sum += sm["msd"]
                hd_sum += sm["hd"]
                assd_sum += sm["assd"]
                count += 1

            if count > 0:
                surface_metrics = {
                    "msd": msd_sum / count,
                    "hd": hd_sum / count,
                    "assd": assd_sum / count
                }

        # 记录历史
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["iou"].append(metrics["iou"])
        history["dice"].append(metrics["dice"])
        history["precision"].append(metrics["precision"])
        history["recall"].append(metrics["recall"])
        history["accuracy"].append(metrics["accuracy"])
        history["msd"].append(surface_metrics["msd"])
        history["hd"].append(surface_metrics["hd"])
        history["assd"].append(surface_metrics["assd"])

        # 保存最佳模型
        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            best_metrics = {
                "epoch": epoch + 1,
                "iou": metrics["iou"],
                "dice": metrics["dice"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "accuracy": metrics["accuracy"],
                "msd": surface_metrics["msd"],
                "hd": surface_metrics["hd"],
                "assd": surface_metrics["assd"]
            }
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

        # 打印进度
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"mIoU: {metrics['iou']:.4f} | "
              f"mDice: {metrics['dice']:.4f} | "
              f"Precision: {metrics['precision']:.4f} | "
              f"Recall: {metrics['recall']:.4f} | "
              f"Time: {epoch_time:.1f}s")

        # 每10个epoch保存一次训练进度图
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            plot_training_progress(history, epoch + 1, output_dir)

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))

    # 保存训练历史
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # 绘制最终训练曲线
    plot_final_training_curves(history, output_dir)

    return history, best_metrics


# 绘制训练进度图
def plot_training_progress(history, epoch, output_dir):
    plt.figure(figsize=(15, 12))

    # 损失曲线（对数坐标）
    plt.subplot(2, 3, 1)
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
    plt.title(f"Loss Curves (Epoch {epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.yscale('log')  # 设置为对数坐标
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # IoU曲线（线性坐标）
    plt.subplot(2, 3, 2)
    plt.plot(history["epoch"], history["iou"], 'g-', label="mIoU")
    plt.title(f"mIoU Progress (Epoch {epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.grid(True)

    # Dice曲线（线性坐标）
    plt.subplot(2, 3, 3)
    plt.plot(history["epoch"], history["dice"], 'm-', label="Dice")
    plt.title(f"Dice Coefficient (Epoch {epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.grid(True)

    # Precision-Recall曲线（线性坐标）
    plt.subplot(2, 3, 4)
    plt.plot(history["epoch"], history["precision"], 'b-', label="Precision")
    plt.plot(history["epoch"], history["recall"], 'r-', label="Recall")
    plt.title(f"Precision & Recall (Epoch {epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # 距离指标曲线（对数坐标）
    plt.subplot(2, 3, 5)
    plt.plot(history["epoch"], history["msd"], 'c-', label="MSD (mm)")
    plt.plot(history["epoch"], history["hd"], 'y-', label="HD (mm)")
    plt.plot(history["epoch"], history["assd"], 'k-', label="ASSD (mm)")
    plt.title("Distance Metrics (mm)")
    plt.xlabel("Epoch")
    plt.ylabel("Distance (mm, log scale)")
    plt.yscale('log')  # 设置为对数坐标
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_progress_epoch_{epoch}.png"))
    plt.close()


# 绘制最终训练曲线（使用对数坐标）
def plot_final_training_curves(history, output_dir):
    plt.figure(figsize=(15, 10))

    # 主损失曲线（对数坐标）
    plt.subplot(2, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], 'b-', label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], 'r-', label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.yscale('log')  # 设置为对数坐标
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # 分割指标曲线（线性坐标）
    plt.subplot(2, 2, 2)
    plt.plot(history["epoch"], history["iou"], 'g-', label="mIoU")
    plt.plot(history["epoch"], history["dice"], 'm-', label="Dice")
    plt.title("Segmentation Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # Precision-Recall曲线（线性坐标）
    plt.subplot(2, 2, 3)
    plt.plot(history["epoch"], history["precision"], 'c-', label="Precision")
    plt.plot(history["epoch"], history["recall"], 'y-', label="Recall")
    plt.title("Precision and Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # 距离指标曲线（对数坐标）
    plt.subplot(2, 2, 4)
    plt.plot(history["epoch"], history["msd"], 'r-', label="MSD (mm)")
    plt.plot(history["epoch"], history["hd"], 'g-', label="HD (mm)")
    plt.plot(history["epoch"], history["assd"], 'b-', label="ASSD (mm)")
    plt.title("Distance Metrics (mm)")
    plt.xlabel("Epoch")
    plt.ylabel("Distance (mm, log scale)")
    plt.yscale('log')  # 设置为对数坐标
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.save(os.path.join(output_dir, "final_training_curves.png"))
    plt.close()


# 主函数 - 支持多个路径
def main():
    # 收集所有图像和掩码路径
    image_paths = []
    mask_paths = []

    print("开始收集数据集...")
    print(f"图像目录数量: {len(original_image_dirs)}")
    print(f"掩码目录数量: {len(original_mask_dirs)}")

    # 确保图像目录和掩码目录数量相同
    if len(original_image_dirs) != len(original_mask_dirs):
        raise ValueError("图像目录和掩码目录数量不一致")

    # 遍历所有路径对
    for img_dir, mask_dir in zip(original_image_dirs, original_mask_dirs):
        print(f"\n处理路径对: 图像目录={img_dir}, 掩码目录={mask_dir}")

        # 获取当前目录下的所有图像和掩码文件
        img_files = glob.glob(os.path.join(img_dir, "*.png"))
        mask_files = glob.glob(os.path.join(mask_dir, "*.png"))

        print(f"  找到 {len(img_files)} 个图像文件和 {len(mask_files)} 个掩码文件")

        # 创建映射关系：数字ID -> 文件路径
        image_dict = {}
        mask_dict = {}

        # 处理图像文件
        for img_path in img_files:
            base_name = os.path.basename(img_path)
            # 支持多种命名格式
            if base_name.startswith("LIDC_") and base_name.endswith(".png"):
                parts = base_name.split("_")
                if len(parts) > 1:
                    num_str = parts[1].split(".")[0]
                    image_dict[num_str] = img_path
            elif base_name.startswith("Image_") and base_name.endswith(".png"):
                parts = base_name.split("_")
                if len(parts) > 1:
                    num_str = parts[1].split(".")[0]
                    image_dict[num_str] = img_path
            else:
                # 尝试从文件名中提取数字ID
                num_str = ''.join(filter(str.isdigit, base_name.split(".")[0]))
                if num_str:
                    image_dict[num_str] = img_path

        # 处理掩码文件
        for mask_path in mask_files:
            base_name = os.path.basename(mask_path)
            # 支持多种命名格式
            if base_name.startswith("LIDC_Mask_") and base_name.endswith(".png"):
                parts = base_name.split("_")
                if len(parts) > 2:
                    num_str = parts[2].split(".")[0]
                    mask_dict[num_str] = mask_path
            elif base_name.startswith("Mask_") and base_name.endswith(".png"):
                parts = base_name.split("_")
                if len(parts) > 1:
                    num_str = parts[1].split(".")[0]
                    mask_dict[num_str] = mask_path
            else:
                # 尝试从文件名中提取数字ID
                num_str = ''.join(filter(str.isdigit, base_name.split(".")[0]))
                if num_str:
                    mask_dict[num_str] = mask_path

        # 匹配具有相同数字ID的文件
        matched_ids = set(image_dict.keys()) & set(mask_dict.keys())
        print(f"  匹配到 {len(matched_ids)} 对图像和掩码")

        if not matched_ids:
            print(f"警告: 在目录对 {img_dir} 和 {mask_dir} 中没有找到匹配的图像和掩码")
            continue

        # 确保顺序一致
        sorted_ids = sorted(matched_ids)
        for num_str in sorted_ids:
            image_paths.append(image_dict[num_str])
            mask_paths.append(mask_dict[num_str])

    print(f"\n总共收集到 {len(image_paths)} 对图像和掩码")

    if len(image_paths) == 0:
        raise ValueError("没有找到任何匹配的图像和掩码文件")

    # 创建数据集验证报告
    create_data_validation_report(image_paths, mask_paths, output_dir)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 五折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_fold_metrics = []

    # 将路径列表转换为numpy数组用于KFold
    image_paths = np.array(image_paths)
    mask_paths = np.array(mask_paths)

    # 保存折叠划分信息
    fold_info = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths)):
        fold_start = time.time()
        print(f"\n=== 训练第 {fold + 1} 折 ===")

        # 创建折叠输出目录
        fold_dir = os.path.join(output_dir, f"fold_{fold + 1}")
        os.makedirs(fold_dir, exist_ok=True)

        # 保存折叠划分信息
        fold_info.append({
            "fold": fold + 1,
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist()
        })

        # 创建数据加载器
        train_dataset = MedicalDataset(
            image_paths[train_idx].tolist(),
            mask_paths[train_idx].tolist()
        )

        val_dataset = MedicalDataset(
            image_paths[val_idx].tolist(),
            mask_paths[val_idx].tolist()
        )

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, pin_memory=True)

        print(f"训练样本数: {len(train_dataset)} | 验证样本数: {len(val_dataset)}")

        # 初始化模型
        model = UNet(in_channels=1, out_channels=1).to(device)

        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")

        # 设置优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()  # 二值交叉熵损失

        # 训练模型（100轮次）
        fold_history, best_metrics = train_model(
            model, train_loader, val_loader,
            optimizer, criterion, device, epochs=100
        )

        # 添加折叠信息和参数量
        best_metrics["fold"] = fold + 1
        best_metrics["parameters"] = total_params
        best_metrics["epoch"] = best_metrics["epoch"]  # 最佳epoch

        # 保存每折的最佳指标
        all_fold_metrics.append(best_metrics)

        # 保存每折的指标
        with open(os.path.join(fold_dir, "fold_metrics.json"), "w") as f:
            json.dump(best_metrics, f, indent=4)

        fold_time = time.time() - fold_start
        print(f"第 {fold + 1} 折训练完成! 用时: {fold_time // 60:.0f}分{fold_time % 60:.0f}秒")

    # 保存折叠划分信息
    with open(os.path.join(output_dir, "fold_info.json"), "w") as f:
        json.dump(fold_info, f, indent=4)

    # 计算并保存所有折的统计信息
    stats_dir = os.path.join(output_dir, "summary_statistics")
    os.makedirs(stats_dir, exist_ok=True)

    # 创建DataFrame用于统计
    metrics_df = pd.DataFrame(all_fold_metrics)

    # 计算统计指标
    stats = {}
    for col in metrics_df.columns:
        if col in ["fold", "parameters", "epoch"]:
            continue
        stats[col] = {
            "Mean": metrics_df[col].mean(),
            "Std": metrics_df[col].std(),
            "Median": metrics_df[col].median(),
            "IQR": metrics_df[col].quantile(0.75) - metrics_df[col].quantile(0.25),
            "Min": metrics_df[col].min(),
            "Max": metrics_df[col].max()
        }

    # 保存统计结果
    metrics_df.to_csv(os.path.join(stats_dir, "all_folds_metrics.csv"), index=False)
    with open(os.path.join(stats_dir, "summary_statistics.json"), "w") as f:
        json.dump(stats, f, indent=4)

    # 创建统计表格
    stats_df = pd.DataFrame(stats).T.reset_index()
    stats_df.columns = ['Metric'] + list(stats.values())[0].keys()

    # 格式化统计表格
    formatted_stats = stats_df.copy()
    for col in stats_df.columns[1:]:
        if "mm" in stats_df['Metric'].iloc[0] or "Metric" in col:
            formatted_stats[col] = formatted_stats[col].apply(lambda x: f"{x:.4f}")
        else:
            formatted_stats[col] = formatted_stats[col].apply(lambda x: f"{x:.2f}")

    # 保存为文本文件
    with open(os.path.join(stats_dir, "summary_statistics.txt"), "w") as f:
        f.write("10项指标的综合统计信息:\n")
        f.write("=" * 90 + "\n")
        for _, row in formatted_stats.iterrows():
            f.write(f"{row['Metric']}:\n")
            f.write(f"  均值: {row['Mean']}")
            f.write(f"  标准差: {row['Std']}")
            f.write(f"  中位数: {row['Median']}")
            f.write(f"  四分位距: {row['IQR']}\n")
        f.write("=" * 90 + "\n\n")

        # 添加说明
        f.write("指标说明:\n")
        f.write("- iou: 平均交并比\n")
        f.write("- dice: Dice相似系数\n")
        f.write("- precision: 精确率\n")
        f.write("- recall: 召回率\n")
        f.write("- accuracy: 准确率\n")
        f.write("- msd: 平均表面距离(毫米)\n")
        f.write("- hd: 豪斯多夫距离(毫米)\n")
        f.write("- assd: 平均对称表面距离(毫米)\n")
        f.write("- parameters: 模型参数量\n")
        f.write("- epoch: 最佳模型对应的训练轮次\n")

    # 创建统计图（使用对数坐标）
    plt.figure(figsize=(15, 10))
    melt_df = pd.melt(metrics_df.drop(["parameters", "epoch", "fold"], axis=1),
                      var_name="Metric", value_name="Value")

    # 分离需要对数坐标的指标
    log_metrics = ['msd', 'hd', 'assd']
    linear_metrics = [m for m in melt_df['Metric'].unique() if m not in log_metrics]

    # 创建对数坐标的箱线图
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    log_df = melt_df[melt_df['Metric'].isin(log_metrics)]
    sns.boxplot(x="Metric", y="Value", data=log_df)
    plt.title("Cross-Validation Distance Metrics Distribution (log scale)")
    plt.yscale('log')  # 对数坐标
    plt.xticks(rotation=45)
    plt.grid(True, which="both", ls="--")

    # 创建线性坐标的箱线图
    plt.subplot(2, 1, 2)
    linear_df = melt_df[melt_df['Metric'].isin(linear_metrics)]
    sns.boxplot(x="Metric", y="Value", data=linear_df)
    plt.title("Cross-Validation Segmentation Metrics Distribution")
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, "metrics_boxplot.png"))
    plt.close()

    # 输出统计结果到控制台
    print("\n" + "=" * 80)
    print("10项指标的综合统计信息:")
    print("=" * 80)
    print(formatted_stats.to_string(index=False))
    print("=" * 80)
    print(f"所有详细指标和统计信息已保存至: {stats_dir}")

    print("\n训练完成! 所有结果保存在:", output_dir)


# 创建数据集验证报告
def create_data_validation_report(image_paths, mask_paths, output_dir):
    report = {
        "total_samples": len(image_paths),
        "dimension_mismatch": 0,
        "mask_value_issues": 0,
        "size_distribution": {}
    }

    size_counts = {}

    for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        try:
            img = imageio.imread(img_path)
            mask = imageio.imread(mask_path)

            # 尺寸检查
            if img.shape[:2] != mask.shape[:2]:
                report["dimension_mismatch"] += 1
                print(f"尺寸不匹配: {img_path} ({img.shape}) vs {mask_path} ({mask.shape})")

            # 值范围检查
            mask_values = np.unique(mask)
            if len(mask_values) > 2 or (0 not in mask_values and 255 not in mask_values):
                report["mask_value_issues"] += 1
                print(f"Mask值异常: {mask_path} - 值范围: {mask_values}")

            # 尺寸分布
            size = f"{img.shape[0]}x{img.shape[1]}"
            size_counts[size] = size_counts.get(size, 0) + 1

        except Exception as e:
            print(f"处理文件时出错: {img_path} | {mask_path} - {str(e)}")

    report["size_distribution"] = size_counts

    # 保存报告
    with open(os.path.join(output_dir, "data_validation_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    print("\n" + "=" * 50)
    print("数据集验证报告:")
    print("=" * 50)
    print(f"总样本: {report['total_samples']}")
    print(
        f"尺寸不匹配样本: {report['dimension_mismatch']} ({(report['dimension_mismatch'] / report['total_samples'] * 100):.1f}%)")
    print(f"Mask值异常: {report['mask_value_issues']}")
    print("尺寸分布:")
    for size, count in report["size_distribution"].items():
        print(f"  {size}: {count}张 ({count / report['total_samples'] * 100:.1f}%)")
    print("=" * 50)


if __name__ == "__main__":
    main()