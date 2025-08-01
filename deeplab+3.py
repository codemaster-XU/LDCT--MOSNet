import datetime
import os
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
from nets.deeplabv3_training import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import binary_erosion
import cv2
from PIL import Image
import random
import re
import time
from scipy.ndimage import distance_transform_edt

# 扩展的评估回调类
class AdvancedEvalCallback:
    def __init__(self, net, input_shape, num_classes, save_dir, device, eval_period=5, pixel_spacing=1.0):
        self.net = net
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.save_dir = save_dir
        self.device = device
        self.eval_period = eval_period
        self.pixel_spacing = pixel_spacing
        os.makedirs(save_dir, exist_ok=True)
        
        # 存储指标历史数据
        self.metrics_history = {
            "mIoU": [], "mPA": [], "mPrecision": [], "mRecall": [],
            "Parameter": [], "Epoch": [],
            "Dice": [], "MSD": [], "HD": [], "ASSD": []
        }
        self.epochs = []
        
    def get_surface_points(self, mask):
        """获取表面点坐标"""
        eroded = binary_erosion(mask)
        surface = mask & ~eroded
        return np.argwhere(surface)
    
    def calculate_surface_distances(self, pred, label):
        """计算表面距离指标"""
        try:
            pred_surface = self.get_surface_points(pred)
            label_surface = self.get_surface_points(label)
            
            if len(pred_surface) == 0 or len(label_surface) == 0:
                return 0.0, 0.0, 0.0
            
            hd = max(
                directed_hausdorff(pred_surface, label_surface)[0],
                directed_hausdorff(label_surface, pred_surface)[0]
            )
            
            dist_pred_to_label = []
            for point in pred_surface:
                dists = np.sqrt(np.sum((label_surface - point)**2, axis=1))
                dist_pred_to_label.append(np.min(dists))
            
            dist_label_to_pred = []
            for point in label_surface:
                dists = np.sqrt(np.sum((pred_surface - point)**2, axis=1))
                dist_label_to_pred.append(np.min(dists))
            
            msd_pred = np.mean(dist_pred_to_label) if dist_pred_to_label else 0.0
            msd_label = np.mean(dist_label_to_pred) if dist_label_to_pred else 0.0
            msd = (msd_pred + msd_label) / 2.0
            assd = msd
            
            hd *= self.pixel_spacing
            msd *= self.pixel_spacing
            assd *= self.pixel_spacing
            
            return msd, hd, assd
        except Exception as e:
            print(f"计算表面距离时出错: {e}")
            return 0.0, 0.0, 0.0

    def calculate_dice(self, pred, label):
        """计算Dice相似系数"""
        intersection = np.logical_and(pred, label).sum()
        return (2. * intersection) / (pred.sum() + label.sum() + 1e-7)

    def calculate_iou(self, pred, label):
        """计算IoU"""
        intersection = np.logical_and(pred, label).sum()
        union = np.logical_or(pred, label).sum()
        return intersection / (union + 1e-7)

    def calculate_pa(self, pred, label):
        """计算像素准确率"""
        return np.mean(pred == label)

    def calculate_precision_recall(self, pred, label):
        """计算精确率和召回率"""
        tp = np.logical_and(pred == 1, label == 1).sum()
        fp = np.logical_and(pred == 1, label == 0).sum()
        fn = np.logical_and(pred == 0, label == 1).sum()
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        return precision, recall

    def on_epoch_end(self, epoch, val_loader):
        if epoch % self.eval_period != 0 and epoch != self.net.module.UnFreeze_Epoch - 1:
            return None
            
        self.net.eval()
        
        metrics = {
            "iou": [], "pa": [], "precision": [], "recall": [],
            "dice": [], "msd": [], "hd": [], "assd": []
        }
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device).float()
                labels = labels.to(self.device).long()
                
                with amp.autocast():
                    outputs = self.net(images)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1).cpu().numpy().astype(np.uint8)
                
                labels = labels.cpu().numpy().astype(np.uint8)
                
                for i in range(images.shape[0]):
                    pred = preds[i]
                    label = labels[i]
                    
                    pred_binary = (pred > 0).astype(np.uint8)
                    label_binary = (label > 0).astype(np.uint8)
                    
                    metrics["iou"].append(self.calculate_iou(pred_binary, label_binary))
                    metrics["pa"].append(self.calculate_pa(pred_binary, label_binary))
                    
                    precision, recall = self.calculate_precision_recall(pred_binary, label_binary)
                    metrics["precision"].append(precision)
                    metrics["recall"].append(recall)
                    
                    metrics["dice"].append(self.calculate_dice(pred_binary, label_binary))
                    
                    msd, hd, assd = self.calculate_surface_distances(pred_binary, label_binary)
                    metrics["msd"].append(msd)
                    metrics["hd"].append(hd)
                    metrics["assd"].append(assd)
        
        avg_metrics = {
            "mIoU": np.nanmean(metrics["iou"]),
            "mPA": np.nanmean(metrics["pa"]),
            "mPrecision": np.nanmean(metrics["precision"]),
            "mRecall": np.nanmean(metrics["recall"]),
            "Dice": np.nanmean(metrics["dice"]),
            "MSD": np.nanmean(metrics["msd"]),
            "HD": np.nanmean(metrics["hd"]),
            "ASSD": np.nanmean(metrics["assd"])
        }
        
        total_params = sum(p.numel() for p in self.net.parameters())
        avg_metrics["Parameter"] = total_params
        avg_metrics["Epoch"] = epoch
        
        for key in avg_metrics:
            if key in self.metrics_history:
                self.metrics_history[key].append(avg_metrics[key])
        self.epochs.append(epoch)
        
        with open(os.path.join(self.save_dir, "metrics.json"), "w") as f:
            json.dump({
                "epochs": self.epochs,
                "metrics": self.metrics_history
            }, f, indent=4)
            
        self.plot_metrics(epoch)
        
        self.net.train()
        return avg_metrics

    def plot_metrics(self, epoch):
        plt.figure(figsize=(15, 15))
        plt.suptitle(f"Metrics at Epoch {epoch}", fontsize=16)
        
        plt.subplot(3, 1, 1)
        seg_metrics = [
            ("mIoU", "IoU"),
            ("mPA", "Pixel Accuracy"),
            ("mPrecision", "Precision"),
            ("mRecall", "Recall"),
            ("Dice", "Dice Coefficient")
        ]
        
        for metric, label in seg_metrics:
            plt.plot(self.epochs, self.metrics_history[metric], label=label)
        plt.title("Segmentation Quality Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        distance_metrics = [
            ("MSD", "Mean Surface Distance (mm)"),
            ("HD", "Hausdorff Distance (mm)"),
            ("ASSD", "Average Symmetric Surface Distance (mm)")
        ]
        
        for metric, label in distance_metrics:
            plt.plot(self.epochs, self.metrics_history[metric], label=label)
        plt.title("Distance Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("Distance (mm)")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        other_metrics = [
            ("Parameter", "Parameter Count"),
            ("Epoch", "Epoch")
        ]
        
        for metric, label in other_metrics:
            plt.plot(self.epochs, self.metrics_history[metric], label=label)
        plt.title("Other Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"metrics_epoch_{epoch}.png"))
        plt.close()
        
    def final_summary(self):
        summary = {}
        for metric, values in self.metrics_history.items():
            if metric not in ["Epoch", "Parameter"]:
                summary[f"{metric}_mean"] = np.mean(values)
                summary[f"{metric}_std"] = np.std(values)
        
        total_params = sum(p.numel() for p in self.net.parameters())
        summary["Parameter"] = total_params
        
        with open(os.path.join(self.save_dir, "final_summary.json"), "w") as f:
            json.dump(summary, f, indent=4)
            
        return summary

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, input_shape, num_classes, original_image_dirs, original_mask_dirs, src_type, label_type, 
                 is_train=True, pixel_spacing=1.0):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.is_train = is_train
        self.pixel_spacing = pixel_spacing
        self.image_paths = []
        self.mask_paths = []
        
        for img_dir, mask_dir in zip(original_image_dirs, original_mask_dirs):
            for folder in os.listdir(img_dir):
                img_folder = os.path.join(img_dir, folder)
                if not os.path.isdir(img_folder):
                    continue
                
                folder_prefix = None
                if "PAT" in folder:
                    match = re.match(r'(PAT\d+)', folder)
                    if match:
                        folder_prefix = match.group(1)
                elif "SBL" in folder:
                    match = re.match(r'(SBL\d+)', folder)
                    if match:
                        folder_prefix = match.group(1)
                
                if not folder_prefix:
                    print(f"无法识别文件夹前缀: {folder}")
                    continue
                
                mask_folder_found = False
                mask_folder_path = None
                for mask_folder in os.listdir(mask_dir):
                    if folder_prefix in mask_folder:
                        mask_folder_path = os.path.join(mask_dir, mask_folder)
                        if os.path.isdir(mask_folder_path):
                            mask_folder_found = True
                            break
                
                if not mask_folder_found:
                    print(f"找不到匹配的mask文件夹: {folder_prefix} in {mask_dir}")
                    continue
                
                for file in os.listdir(img_folder):
                    if file.endswith(src_type):
                        img_path = os.path.join(img_folder, file)
                        mask_file = file.replace(src_type, label_type)
                        mask_path = os.path.join(mask_folder_path, mask_file)
                        
                        if os.path.exists(mask_path):
                            self.image_paths.append(img_path)
                            self.mask_paths.append(mask_path)
                        else:
                            print(f"找不到mask文件: {mask_path}")
        
        total_count = len(self.image_paths)
        if total_count == 0:
            raise ValueError("没有找到匹配的图像和掩码文件对!")
            
        indices = list(range(total_count))
        random.shuffle(indices)
        split = int(0.8 * total_count)
        
        if is_train:
            self.indices = indices[:split]
        else:
            self.indices = indices[split:]
            
        print(f"加载了 {len(self.indices)} 个{'训练' if is_train else '验证'}样本")
        print(f"总样本数: {total_count}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        mask = Image.open(mask_path)
        mask = mask.convert('L')
        
        image = np.array(image, dtype=np.uint8)
        mask = np.array(mask, dtype=np.uint8)
        
        mask = mask % self.num_classes
        
        if self.is_train and random.random() < 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        if self.is_train and random.random() < 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)
            
        image = image / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        image = cv2.resize(image, self.input_shape)
        mask = cv2.resize(mask, self.input_shape, interpolation=cv2.INTER_NEAREST)
        
        image = image.transpose(2, 0, 1)
        
        image = image.astype(np.float32)
        mask = mask.astype(np.int64)
        
        return image, mask

# 根据图1重新实现损失函数
class GeneralizedDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, pred, target):
        # 关键修复1：确保pred和target尺寸匹配
        if pred.shape[2:] != target.shape[1:]:
            # 调整target尺寸以匹配pred
            target = F.interpolate(target.unsqueeze(1).float(), size=pred.shape[2:], mode='nearest').squeeze(1).long()
        
        with amp.autocast():
            pred = F.softmax(pred, dim=1)
            
            # 将真实标签转换为one-hot编码
            target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
            
            # 计算每个类别的权重 (1/(体积_c)^2)
            volumes = target_onehot.sum((0, 2, 3))
            weights = 1. / (volumes.float()**2 + self.epsilon)
            
            # 计算分子
            numerator = (pred * target_onehot).sum((0, 2, 3))
            numerator = torch.sum(weights * numerator)
            
            # 计算分母
            denominator_pred = pred.sum((0, 2, 3))
            denominator_target = target_onehot.sum((0, 2, 3))
            denominator = torch.sum(weights * (denominator_pred + denominator_target))
            
            # 计算Dice系数
            dice = 2. * numerator / (denominator + self.epsilon)
            
            return 1. - dice

class BoundaryLoss(nn.Module):
    def __init__(self, pixel_spacing=1.0):
        super(BoundaryLoss, self).__init__()
        self.pixel_spacing = pixel_spacing
        
    def forward(self, pred, target):
        if pred.shape[2:] != target.shape[1:]:
            # 调整target尺寸以匹配pred
            target = F.interpolate(target.unsqueeze(1).float(), size=pred.shape[2:], mode='nearest').squeeze(1).long()
        
        binary_target = (target > 0).float()
        
        with torch.no_grad():
            dist_map = np.zeros_like(binary_target.cpu().numpy())
            for i in range(dist_map.shape[0]):
                for j in range(dist_map.shape[1]):
                    bin_img = binary_target[i, j].cpu().numpy()
                    outer = distance_transform_edt(1 - bin_img) * self.pixel_spacing
                    inner = distance_transform_edt(bin_img) * self.pixel_spacing
                    dist_map[i, j] = outer - inner
        
        dist_map = torch.tensor(dist_map, dtype=torch.float32).to(pred.device)
        
        pred_prob = torch.softmax(pred, dim=1)
        foreground_prob = 1 - pred_prob[:, 0]
        
        boundary_loss = torch.mean(foreground_prob * dist_map)
        
        return boundary_loss

class MultiScaleContextWeightLoss(nn.Module):
    def __init__(self, scales=[1, 2, 4], weights=[0.5, 0.3, 0.2]):
        super(MultiScaleContextWeightLoss, self).__init__()
        self.scales = scales
        self.weights = weights
        
    def forward(self, pred, target):
        loss = 0
        for scale, weight in zip(self.scales, self.weights):
            down_pred = F.avg_pool2d(pred, kernel_size=scale, stride=scale)
            down_target = F.avg_pool2d(target.float().unsqueeze(1), kernel_size=scale, stride=scale).squeeze(1).long()
            
            ce_loss = F.cross_entropy(down_pred, down_target, reduction='mean')
            loss += weight * ce_loss
        
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, pixel_spacing=1.0, lambda_boundary=1.0, lambda_conf=0.5):
        super(CombinedLoss, self).__init__()
        self.gdice_loss = GeneralizedDiceLoss()
        self.boundary_loss = BoundaryLoss(pixel_spacing)
        self.mscw_loss = MultiScaleContextWeightLoss()
        self.lambda_boundary = lambda_boundary
        self.lambda_conf = lambda_conf
        
    def forward(self, pred, target):
        # 关键修复2：确保所有损失组件使用相同的空间尺寸
        if pred.shape[2:] != target.shape[1:]:
            # 调整target尺寸以匹配pred
            target_resized = F.interpolate(target.unsqueeze(1).float(), size=pred.shape[2:], mode='nearest').squeeze(1).long()
        else:
            target_resized = target
            
        dice_loss = self.gdice_loss(pred, target_resized)
        boundary_loss = self.boundary_loss(pred, target_resized)
        mscw_loss = self.mscw_loss(pred, target_resized)
        
        combined_loss = dice_loss + \
                       self.lambda_boundary * boundary_loss + \
                       self.lambda_conf * mscw_loss
                       
        return combined_loss, dice_loss, boundary_loss, mscw_loss

# 自定义的fit_one_epoch函数
def custom_fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, 
                         gen, gen_val, Epoch, cuda, save_period, save_dir, local_rank):
    total_loss = 0
    val_loss = 0
    
    combined_loss = CombinedLoss(pixel_spacing=1.0, lambda_boundary=1.0, lambda_conf=0.5)
    combined_loss = combined_loss.to(cuda)
    
    accumulation_steps = 4
    scaler = amp.GradScaler()
    
    model_train.train()
    
    for iteration, batch in enumerate(gen):
        images, masks = batch
        
        images = images.to(cuda).float()
        masks = masks.to(cuda).long()
        
        with amp.autocast():
            outputs = model_train(images)
            
            # 关键修复3：确保损失计算使用正确的目标尺寸
            loss, dice_loss, boundary_loss, mscw_loss = combined_loss(outputs, masks)
            loss = loss / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (iteration + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
        
        total_loss += loss.item() * accumulation_steps
        
        if iteration % 100 == 0:
            print(f"Epoch: {epoch+1}/{Epoch}, Iteration: {iteration}/{epoch_step}, Loss: {loss.item()*accumulation_steps:.4f}, "
                  f"Dice: {dice_loss.item():.4f}, Boundary: {boundary_loss.item():.4f}, "
                  f"MSCW: {mscw_loss.item():.4f}")
    
    model_train.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(gen_val):
            images, masks = batch
            images = images.to(cuda).float()
            masks = masks.to(cuda).long()
            
            with amp.autocast():
                outputs = model_train(images)
                loss, _, _, _ = combined_loss(outputs, masks)
            val_loss += loss.item()
    
    avg_loss = total_loss / epoch_step
    avg_val_loss = val_loss / epoch_step_val
    
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model_train.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, os.path.join(save_dir, f"epoch_{epoch+1}.pth"))
    
    return avg_loss, avg_val_loss

# ====================== 修改的DeepLabv3+模型 ======================
class DeepLab(nn.Module):
    def __init__(self, num_classes=21, backbone="mobilenet", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        if backbone == "xception":
            self.backbone = self._build_xception_backbone(downsample_factor, pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone == "mobilenet":
            self.backbone = self._build_mobilenet_backbone(downsample_factor, pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.aspp = ASPP(in_channels, 256, [6, 12, 18])
        
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.cat_conv = nn.Sequential(
            nn.Conv2d(256+48, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )
        
        # 关键修复4：使用动态上采样替代固定比例上采样
        # 删除固定比例的上采样初始化
        # 将在forward中动态上采样到原始尺寸

    def forward(self, x):
        # 保存原始图像尺寸
        H, W = x.size(2), x.size(3)
        
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        
        low_level_size = low_level_features.shape[2:]
        x = F.interpolate(x, size=low_level_size, mode='bilinear', align_corners=True)
        
        low_level_features = self.shortcut_conv(low_level_features)
        
        x = torch.cat((x, low_level_features), dim=1)
        x = self.cat_conv(x)
        
        # 关键修复5：动态上采样到原始输入尺寸
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

    def _build_xception_backbone(self, downsample_factor, pretrained):
        return XceptionBackbone(downsample_factor, pretrained)
    
    def _build_mobilenet_backbone(self, downsample_factor, pretrained):
        return MobileNetBackbone(downsample_factor, pretrained)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        modules = []
        
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        for r in rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(modules)*out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            if isinstance(conv[0], nn.AdaptiveAvgPool2d):
                y = conv(x)
                y = F.interpolate(y, size=x.size()[2:], mode='bilinear', align_corners=True)
                res.append(y)
            else:
                res.append(conv(x))
        
        x = torch.cat(res, dim=1)
        return self.project(x)

# 简化的主干网络实现
class XceptionBackbone(nn.Module):
    def __init__(self, downsample_factor, pretrained):
        super(XceptionBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        self.low_level_conv = nn.Conv2d(64, 256, kernel_size=1)
        self.high_level_conv = nn.Conv2d(64, 2048, kernel_size=1)
        
        if downsample_factor == 16:
            self.stride = 16

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        low_level_feat = self.low_level_conv(x)
        x = self.high_level_conv(x)
        return low_level_feat, x

class MobileNetBackbone(nn.Module):
    def __init__(self, downsample_factor, pretrained):
        super(MobileNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU6()
        
        self.low_level_conv = nn.Conv2d(32, 24, kernel_size=1)
        self.high_level_conv = nn.Conv2d(32, 320, kernel_size=1)
        
        if downsample_factor == 16:
            self.stride = 16

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        low_level_feat = self.low_level_conv(x)
        x = self.high_level_conv(x)
        return low_level_feat, x
# ====================== 修改结束 ======================

if __name__ == "__main__":
    # =================== 用户配置区域 =================== 
    original_image_dirs = [
        "XJTU/nii Translatepng/outputfile",
        "XJTU/nii Translatepng/outputfile_sbl",
        "XJTU/nii Translatepng/outputfile1"
    ]
    original_mask_dirs = [
        "XJTU/nii Translatepng/outputfile_mask",
        "XJTU/nii Translatepng/outputfile_sbl_mask",
        "XJTU/nii Translatepng/outputfile1_mask"
    ]
    
    pixel_spacing = 1.0
    num_classes = 57
    target_size = (512, 512)
    batch_size = 2
    src_type = '.png'
    label_type = '.png'
    
    output_dir = "deeplabV3+3-output"
    os.makedirs(output_dir, exist_ok=True)
    
    Init_Epoch = 0
    Freeze_Epoch = 0
    UnFreeze_Epoch = 100
    Freeze_Train = False
    
    backbone = "mobilenet"
    pretrained = True
    model_path = ""
    downsample_factor = 16
    input_shape = target_size
    
    Init_lr = 7e-4
    Min_lr = Init_lr * 0.01
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0
    lr_decay_type = 'cos'
    
    save_period = 5
    save_dir = 'logs'
    eval_flag = True
    eval_period = 5
    dice_loss = False
    focal_loss = False
    cls_weights = np.ones([num_classes], np.float32)
    num_workers = 4
    Cuda = True
    seed = 11
    distributed = False
    sync_bn = False
    fp16 = True
    
    seed_everything(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = 0

    model = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor,
                    pretrained=pretrained)
    
    if model_path != '':
        print(f'加载权重从 {model_path}')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    print("加载训练数据集...")
    train_dataset = CustomDataset(
        input_shape=input_shape, 
        num_classes=num_classes,
        original_image_dirs=original_image_dirs,
        original_mask_dirs=original_mask_dirs,
        src_type=src_type,
        label_type=label_type,
        is_train=True,
        pixel_spacing=pixel_spacing
    )
    
    print("加载验证数据集...")
    val_dataset = CustomDataset(
        input_shape=input_shape, 
        num_classes=num_classes,
        original_image_dirs=original_image_dirs,
        original_mask_dirs=original_mask_dirs,
        src_type=src_type,
        label_type=label_type,
        is_train=False,
        pixel_spacing=pixel_spacing
    )
    
    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, 
                     num_workers=num_workers, pin_memory=True, drop_last=True)
    gen_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, 
                         num_workers=num_workers, pin_memory=True, drop_last=True)
    
    model_train = model.to(device)
    
    eval_callback = AdvancedEvalCallback(
        model_train, input_shape, num_classes, 
        output_dir, device, eval_period=eval_period,
        pixel_spacing=pixel_spacing
    )
    
    optimizer = optim.Adam(model.parameters(), lr=Init_lr, weight_decay=weight_decay)
    
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, UnFreeze_Epoch)
    
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        
        train_loss, val_loss = custom_fit_one_epoch(
            model_train, model, None, eval_callback, 
            optimizer, epoch, len(gen), len(gen_val), 
            gen, gen_val, UnFreeze_Epoch, device, 
            save_period, save_dir, local_rank
        )
                      
        print(f"Epoch {epoch+1}/{UnFreeze_Epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                      
        if eval_flag and (epoch % eval_period == 0 or epoch == UnFreeze_Epoch - 1):
            metrics = eval_callback.on_epoch_end(epoch, gen_val)
            if metrics:
                print(f"Epoch {epoch+1}/{UnFreeze_Epoch} Metrics: {metrics}")
                
    summary = eval_callback.final_summary()
    print("最终指标总结:")
    print(json.dumps(summary, indent=4))
    
    final_metrics = {
        "Epochs": UnFreeze_Epoch,
        "Parameter": summary.get("Parameter", 0),
        "mIoU_mean": summary.get("mIoU_mean", 0),
        "mIoU_std": summary.get("mIoU_std", 0),
        "mPA_mean": summary.get("mPA_mean", 0),
        "mPA_std": summary.get("mPA_std", 0),
        "mPrecision_mean": summary.get("mPrecision_mean", 0),
        "mPrecision_std": summary.get("mPrecision_std", 0),
        "mRecall_mean": summary.get("mRecall_mean", 0),
        "mRecall_std": summary.get("mRecall_std", 0),
        "Dice_mean": summary.get("Dice_mean", 0),
        "Dice_std": summary.get("Dice_std", 0),
        "MSD_mean": summary.get("MSD_mean", 0),
        "MSD_std": summary.get("MSD_std", 0),
        "HD_mean": summary.get("HD_mean", 0),
        "HD_std": summary.get("HD_std", 0),
        "ASSD_mean": summary.get("ASSD_mean", 0),
        "ASSD_std": summary.get("ASSD_std", 0),
    }
    
    with open(os.path.join(output_dir, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=4)

    print("训练完成! 总轮次:", UnFreeze_Epoch)