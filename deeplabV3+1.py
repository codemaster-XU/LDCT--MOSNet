import os
import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim import lr_scheduler
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import json
import time
import copy
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from collections import Counter
import concurrent.futures
import multiprocessing
import cv2

# ====================== 高性能配置 ======================
class Config:
    # 设备配置（针对RTX 4090优化）
    device = torch.device("cuda")
    seed = 42
    
    # 数据路径
    root_dir = "XJTU/nii Translatepng/"
    img_type = ".png"
    split_ratio = 0.8
    input_size = (512, 512)
    
    # 医学影像专用参数
    num_classes = 11  # 背景(0)+10个器官
    top_k_organs = 10
    is_grayscale = True  # 标记为单通道图像
    
    # 模型参数
    backbone = "resnet101"
    output_stride = 16
    aspp_rates = [6, 12, 18]
    
    # 训练参数（充分利用24GB显存）
    batch_size = 32  # 进一步增大batch_size
    learning_rate = 3e-4
    epochs = 100
    save_dir = "deeplabV3-ct-optimized"
    checkpoint_freq = 5
    warmup_epochs = 5
    
    # GPU加速设置
    amp = True
    num_workers = multiprocessing.cpu_count()  # 自动获取CPU核心数
    
    # 单通道CT预处理
    ct_window = (-500, 500)  # CT值截断窗口
    normalize_mean = [0.5]    # 单通道专用归一化参数
    normalize_std = [0.5]
    
    # 标签统计参数（优化关键）
    label_stat_chunk_size = 1000  # 每个任务块处理的文件数
    label_stat_sample_pixels = 500  # 每张mask采样像素数
    max_workers = 8  # 并行进程数
    
    # 评估指标
    metrics = ['mIoU', 'mPA', 'mPrecision', 'mRecall', 'Dice', 
               'MSD', 'HD', 'ASSD', 'Epoch', 'Parameters',
               'OrganAreas']  # 添加器官面积统计指标

# 设置随机种子和CUDA优化
torch.manual_seed(Config.seed)
np.random.seed(Config.seed)
random.seed(Config.seed)
torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed_all(Config.seed)

# ====================== 高速标签统计和器官面积分析 ======================
def fast_label_statistics_and_area_calculation():
    print(f"启动高速标签统计和器官面积分析（进程数：{Config.max_workers}）...")
    start_time = time.time()
    
    # 1. 收集所有mask文件路径
    mask_paths = []
    image_mask_mapping = {
        "outputfile": "outputfile_mask",
        "outputfile_sbl": "outputfile_sbl_mask",
        "outputfile1": "outputfile1_mask"
    }
    
    for img_dir, mask_dir in image_mask_mapping.items():
        mask_base = os.path.join(Config.root_dir, mask_dir)
        if not os.path.exists(mask_base):
            continue
            
        case_dirs = [d for d in os.listdir(mask_base) 
                    if os.path.isdir(os.path.join(mask_base, d))]
        
        for case_dir in case_dirs:
            case_path = os.path.join(mask_base, case_dir)
            mask_paths.extend([
                os.path.join(case_path, f) 
                for f in os.listdir(case_path) 
                if f.endswith(Config.img_type)
            ])
    
    print(f"发现{len(mask_paths)}个mask文件")
    if not mask_paths:
        print("警告：未发现mask文件，请检查路径配置！")
        return {}
    
    # 2. 并行批处理标签统计和器官面积计算
    label_counter = Counter()
    area_accumulator = {}  # 用于器官面积累加
    total_chunks = (len(mask_paths) // Config.label_stat_chunk_size + 1)
    
    def process_mask_chunk(chunk_paths):
        """处理一组mask文件并返回局部统计和器官面积"""
        local_counter = Counter()
        local_area = {}
        
        for path in chunk_paths:
            try:
                # 高效加载mask
                with Image.open(path) as img:
                    arr = np.array(img).astype(np.uint8)
                    
                    # 随机采样像素点
                    if arr.size > Config.label_stat_sample_pixels:
                        sampled_pixels = np.random.choice(
                            arr.ravel(), 
                            Config.label_stat_sample_pixels, 
                            replace=False
                        )
                    else:
                        sampled_pixels = arr.ravel()
                    local_counter.update(sampled_pixels)
                    
                    # 计算每个标签的实际像素数（器官面积）
                    unique_labels, counts = np.unique(arr, return_counts=True)
                    for label, count in zip(unique_labels, counts):
                        local_area.setdefault(label, 0)
                        local_area[label] += count
                        
            except Exception as e:
                print(f"处理文件{path}时出错: {str(e)}")
        return local_counter, local_area
    
    # 分块处理
    chunks = [
        mask_paths[i:i+Config.label_stat_chunk_size] 
        for i in range(0, len(mask_paths), Config.label_stat_chunk_size)
    ]
    
    # 并行处理
    results = []
    with tqdm(total=len(chunks), desc="标签统计与器官面积计算") as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=Config.max_workers) as executor:
            futures = {executor.submit(process_mask_chunk, chunk): chunk for chunk in chunks}
            for future in concurrent.futures.as_completed(futures):
                chunk_counter, chunk_area = future.result()
                results.append((chunk_counter, chunk_area))
                pbar.update(1)
    
    # 合并结果
    for chunk_counter, chunk_area in results:
        label_counter.update(chunk_counter)
        for label, area in chunk_area.items():
            area_accumulator.setdefault(label, 0)
            area_accumulator[label] += area
    
    # 3. 构建标签映射
    print(f"统计完成，共发现{len(label_counter)}种标签")
    
    # 排除背景0后取前10高频标签
    non_zero_labels = [label for label in label_counter if label != 0]
    top_labels = sorted(non_zero_labels, key=lambda x: label_counter[x], reverse=True)[:Config.top_k_organs]
    
    # 构建标签映射表 (0:背景, 1-10:目标器官)
    label_mapping = {0: 0}  # 背景固定为0
    area_percentages = []  # 存储每个器官的面积百分比
    
    for idx, label in enumerate(top_labels, 1):
        label_mapping[label] = idx
        total_area = sum(area_accumulator.values())
        organ_area = area_accumulator.get(label, 0)
        area_percentage = organ_area / total_area * 100
        area_percentages.append(area_percentage)
    
    # 添加未识别标签到背景
    for label in non_zero_labels:
        if label not in top_labels:
            label_mapping[label] = 0
    
    # 保存器官面积信息
    organ_info = []
    for i, (label, mapped_label) in enumerate(label_mapping.items()):
        if mapped_label == 0:
            continue
        organ_area = area_accumulator.get(label, 0)
        organ_info.append({
            "original_label": int(label),
            "mapped_label": int(mapped_label),
            "pixel_count": int(organ_area),
            "relative_percentage": area_percentages[i-1]
        })
    
    # 按面积排序器官信息
    organ_info_sorted = sorted(organ_info, key=lambda x: x["relative_percentage"], reverse=True)
    
    print(f"前{Config.top_k_organs}高频器官标签（按面积统计）：")
    for org in organ_info_sorted:
        print(f"  器官{org['mapped_label']} (原标签{org['original_label']}): "
              f"像素数{org['pixel_count']:,} "
              f"面积占比{org['relative_percentage']:.2f}%")
    
    # 可视化器官面积分布
    visualize_organ_areas(organ_info_sorted)
    
    # 保存标签映射和器官信息
    os.makedirs(Config.save_dir, exist_ok=True)
    mapping_path = os.path.join(Config.save_dir, 'label_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(label_mapping, f)
        print(f"标签映射表已保存至 {mapping_path}")
    
    organ_info_path = os.path.join(Config.save_dir, 'organ_areas.json')
    with open(organ_info_path, 'w') as f:
        json.dump(organ_info_sorted, f, indent=4)
        print(f"器官面积信息已保存至 {organ_info_path}")
    
    # 性能报告
    total_time = time.time() - start_time
    print(f"标签统计和器官面积计算完成! 耗时: {total_time:.1f}秒")
    
    return label_mapping, organ_info_sorted

def visualize_organ_areas(organ_info):
    """可视化器官面积分布"""
    plt.figure(figsize=(12, 8))
    
    # 提取器官标签和面积占比
    labels = [f"器官{org['mapped_label']} (原{org['original_label']})" for org in organ_info]
    percentages = [org['relative_percentage'] for org in organ_info]
    
    # 创建条形图
    bars = plt.barh(labels, percentages, color='skyblue')
    plt.xlabel('面积占比 (%)')
    plt.title('各器官面积分布')
    plt.xlim(0, max(percentages) * 1.1)
    
    # 添加数据标签
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, 
                 f'{width:.2f}%', 
                 ha='left', va='center')
    
    plt.tight_layout()
    
    # 保存图表
    area_plot_path = os.path.join(Config.save_dir, 'organ_area_distribution.png')
    plt.savefig(area_plot_path)
    print(f"器官面积分布图已保存至 {area_plot_path}")
    plt.close()

# ====================== 单通道CT数据集加载 ======================
class CTSegmentationDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        self.root_dir = root_dir
        self.mode = mode
        self.samples = []
        self.image_mask_mapping = {
            "outputfile": "outputfile_mask",
            "outputfile_sbl": "outputfile_sbl_mask",
            "outputfile1": "outputfile1_mask"
        }
        self.label_mapping = self.load_label_mapping()
        self._prepare_dataset()
        
    def load_label_mapping(self):
        mapping_path = os.path.join(Config.save_dir, 'label_mapping.json')
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                return json.load(f)
        return None
    
    def _prepare_dataset(self):
        for img_dir, mask_dir in self.image_mask_mapping.items():
            img_base = os.path.join(self.root_dir, img_dir)
            mask_base = os.path.join(self.root_dir, mask_dir)
            
            if not os.path.exists(img_base) or not os.path.exists(mask_base):
                continue
                
            case_dirs = [d for d in os.listdir(img_base) 
                        if os.path.isdir(os.path.join(img_base, d))]
            random.shuffle(case_dirs)
            
            split_idx = int(len(case_dirs) * Config.split_ratio)
            if self.mode == 'train':
                selected_cases = case_dirs[:split_idx]
            else:
                selected_cases = case_dirs[split_idx:]
            
            for case_dir in tqdm(selected_cases, desc=f'加载 {self.mode} 数据 ({img_dir})'):
                case_path = os.path.join(img_base, case_dir)
                
                # 提取病例ID
                if "PAT" in case_dir:
                    case_id = re.split(r'_', case_dir)[0]
                    mask_case_dir = f"{case_id}_merged"
                elif "SBL" in case_dir:
                    case_id = case_dir
                    mask_case_dir = f"{case_id}_merged"
                else:
                    continue
                
                mask_case_path = os.path.join(mask_base, mask_case_dir)
                if not os.path.exists(mask_case_path):
                    continue
                
                # 匹配所有切片
                slice_files = [f for f in os.listdir(case_path) 
                              if f.endswith(Config.img_type)]
                for img_name in slice_files:
                    img_path = os.path.join(case_path, img_name)
                    mask_path = os.path.join(mask_case_path, img_name)
                    
                    if os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path))
    
    def __len__(self):
        return len(self.samples)
    
    def _preprocess_ct(self, img_array):
        """CT图像专用预处理"""
        # CT值截断
        img_array = np.clip(img_array, *Config.ct_window)
        # 标准化
        min_val, max_val = Config.ct_window
        img_array = (img_array - min_val) / (max_val - min_val)
        return img_array
    
    def apply_label_mapping(self, mask_array):
        """应用标签映射"""
        if self.label_mapping is None:
            return mask_array
        
        # 创建映射后的mask
        remapped = np.zeros_like(mask_array, dtype=np.uint8)
        
        # 应用映射
        for orig_label, new_label in self.label_mapping.items():
            remapped[mask_array == orig_label] = new_label
            
        return remapped
    
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        
        # 单通道CT图像加载
        img = Image.open(img_path).convert('L')
        img_array = np.array(img)
        
        # CT专用预处理
        img_array = self._preprocess_ct(img_array)
        img = Image.fromarray(img_array)
        
        # 加载mask
        mask = Image.open(mask_path).convert('L')
        mask_array = np.array(mask)
        
        # 应用标签映射
        mask_array = self.apply_label_mapping(mask_array)
        mask = Image.fromarray(mask_array)
        
        # 数据增强（训练集）
        if self.mode == 'train':
            # 随机水平/垂直翻转
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            
            # 随机旋转（-15°到15°）
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, resample=Image.BICUBIC)
            mask = mask.rotate(angle, resample=Image.NEAREST)
            
            # 随机调整窗宽窗位（CT专用增强）
            if random.random() > 0.5:
                center = random.uniform(-100, 100)
                width = random.uniform(300, 700)
                min_val = max(np.min(img_array), center - width/2)
                max_val = min(np.max(img_array), center + width/2)
                win_img = np.clip(img_array, min_val, max_val)
                win_img = (win_img - min_val) / (max_val - min_val + 1e-8)
                img = Image.fromarray(win_img)
        
        # 调整尺寸
        img = img.resize(Config.input_size, Image.BICUBIC)
        mask = mask.resize(Config.input_size, Image.NEAREST)
        
        # 转换为张量
        img_tensor = transforms.ToTensor()(img)  # 形状 [1, H, W]
        
        # 为兼容预训练模型转换为三通道（复制单通道）
        if Config.is_grayscale:
            img_tensor = img_tensor.repeat(3, 1, 1)  # [3, H, W]
        
        # 单通道专用归一化
        normalize = transforms.Normalize(
            mean=Config.normalize_mean * 3,  # 三通道相同均值
            std=Config.normalize_std * 3     # 三通道相同标准差
        )
        img_tensor = normalize(img_tensor)
        
        mask_tensor = torch.from_numpy(np.array(mask)).long()
        
        return img_tensor, mask_tensor

# ====================== DeepLabV3+ with DenseASPP 模型 ======================
class DenseASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, rates=[3, 6, 12, 18]):
        super(DenseASPP, self).__init__()
        
        # 初始1x1卷积降维
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 全局池化分支
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # DenseASPP块定义
        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()
        self.growth_rate = 32
        
        # 创建连续的DenseASPP块
        for i, dilation in enumerate(rates):
            block = nn.Sequential(
                nn.Conv2d(128 + i * self.growth_rate, self.growth_rate, 
                          kernel_size=3, padding=dilation, dilation=dilation,
                          bias=False),
                nn.BatchNorm2d(self.growth_rate),
                nn.ReLU(inplace=True)
            )
            self.dense_blocks.append(block)
            
            # 每个块后的过渡层（1x1卷积降维）
            transition = nn.Sequential(
                nn.Conv2d(128 + (i+1) * self.growth_rate, 128, 
                         kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
            self.transition_layers.append(transition)
            
        # 特征融合层
        self.final_conv = nn.Sequential(
            nn.Conv2d(128 + len(rates) * self.growth_rate + 64, out_channels, 
                     kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        # 初始特征提取
        x = self.init_conv(x)
        init_feat = x
        
        # 全局池化分支处理
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:], 
                                   mode='bilinear', align_corners=True)
        
        # DenseASPP特征提取
        dense_features = [x]
        for i, (block, transition) in enumerate(zip(self.dense_blocks, self.transition_layers)):
            # 将所有先前的特征连接作为输入
            current_input = torch.cat(dense_features, dim=1) if i > 0 else x
            
            # 通过当前块处理
            out = block(current_input)
            dense_features.append(out)
            
            # 通过过渡层处理当前块输出
            transition_input = torch.cat(dense_features, dim=1)
            transition_out = transition(transition_input)
            
            # 更新特征列表：保留初始特征和最新的过渡输出
            dense_features = [dense_features[0], transition_out]
        
        # 将所有特征和全局特征连接
        features = torch.cat(dense_features, dim=1)
        all_features = torch.cat([features, global_feat], dim=1)
        
        # 最终卷积融合
        return self.final_conv(all_features)

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, backbone='resnet101', output_stride=16):
        super(DeepLabV3Plus, self).__init__()
        if backbone == 'resnet101':
            resnet = models.resnet101(pretrained=True)
        else:
            resnet = models.resnet50(pretrained=True)
            
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # 修改输出步长
        if output_stride == 8:
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding = (2, 2), (2, 2)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        
        # 使用DenseASPP代替原始ASPP
        self.aspp = DenseASPP(2048, 256)  # 输入通道2048，输出256
        self.reduce = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        
        self.upsample4x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        if Config.is_grayscale:
            x = x[:, :1, :, :]  # 只使用第一个通道（兼容单通道输入）
        
        # 编码器
        x = self.layer0(x)       # [N,64,H/4,W/4]
        x1 = self.layer1(x)       # [N,256,H/4,W/4]
        x2 = self.layer2(x1)       # [N,512,H/8,W/8]
        x3 = self.layer3(x2)       # [N,1024,H/16,W/16]
        x4 = self.layer4(x3)       # [N,2048,H/32,W/32]
        
        # ASPP处理 (DenseASPP)
        aspp_out = self.aspp(x4)  # [N,256,H/32,W/32]
        aspp_out = self.upsample4x(aspp_out)  # 上采样至H/8
        
        # 浅层特征
        low_feat = self.reduce(x1)  # [N,48,H/4,W/4]
        
        # 特征融合
        if aspp_out.shape[2:] != low_feat.shape[2:]:
            aspp_out = self.upsample2x(aspp_out)  # 上采样至H/4
            aspp_out = F.interpolate(aspp_out, size=low_feat.shape[2:], 
                                     mode='bilinear', align_corners=True)
        
        fused = torch.cat([aspp_out, low_feat], dim=1)  # [N,304,H/4,W/4]
        
        # 解码器
        out = self.decoder(fused)
        out = self.upsample4x(out)  # 输出原图尺寸
        
        return out

# ====================== 训练流程 ======================
def train_ct_model():
    # 创建输出目录
    os.makedirs(Config.save_dir, exist_ok=True)
    
    # 优先执行高速标签统计和器官面积分析
    if not os.path.exists(os.path.join(Config.save_dir, 'label_mapping.json')):
        print("执行高速标签统计和器官面积分析...")
        label_mapping, organ_info = fast_label_statistics_and_area_calculation()
        print("标签统计和器官面积分析完成！")
    else:
        print("检测到已存在的标签映射表，跳过统计步骤")
        with open(os.path.join(Config.save_dir, 'label_mapping.json'), 'r') as f:
            label_mapping = json.load(f)
        with open(os.path.join(Config.save_dir, 'organ_areas.json'), 'r') as f:
            organ_info = json.load(f)
    
    # 初始化数据集
    train_dataset = CTSegmentationDataset(Config.root_dir, 'train')
    val_dataset = CTSegmentationDataset(Config.root_dir, 'val')
    
    print(f"训练集切片: {len(train_dataset)} | 验证集切片: {len(val_dataset)}")
    
    # 高性能数据加载
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True
    )
    
    # 初始化单通道CT专用模型
    model = DeepLabV3Plus(Config.num_classes, Config.backbone, Config.output_stride).to(Config.device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params/1e6:.2f}M")
    
    # 优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=3, verbose=True)
    
    # 混合精度训练
    scaler = GradScaler(enabled=Config.amp)
    
    # 记录器
    history = {metric: [] for metric in Config.metrics}
    best_iou = 0.0
    
    # 训练循环
    for epoch in range(Config.epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"训练 Epoch {epoch+1}/{Config.epochs}")
        
        for images, masks in progress:
            images = images.to(Config.device, non_blocking=True)
            masks = masks.to(Config.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # 混合精度前向
            with autocast(enabled=Config.amp):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            # 混合精度反向
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(train_loader)
        
        # 验证阶段
        val_metrics = evaluate_ct(model, val_loader, Config.device)
        scheduler.step(val_metrics['mIoU'])
        
        # 记录指标
        history['Epoch'].append(epoch+1)
        history['Parameters'].append(total_params)
        for metric in ['mIoU', 'mPA', 'mPrecision', 'mRecall', 'Dice', 'MSD', 'HD', 'ASSD']:
            history[metric].append(val_metrics[metric])
        
        # 添加器官面积信息（首次epoch添加）
        if epoch == 0:
            with open(os.path.join(Config.save_dir, 'organ_areas.json'), 'r') as f:
                organ_info = json.load(f)
            history['OrganAreas'] = organ_info
        
        # 保存最佳模型
        if val_metrics['mIoU'] > best_iou:
            best_iou = val_metrics['mIoU']
            torch.save(model.state_dict(), os.path.join(Config.save_dir, "best_ct_model.pth"))
            print(f"保存最佳模型 mIoU: {best_iou:.4f}")
        
        # 定期保存
        if (epoch+1) % Config.checkpoint_freq == 0 or epoch == Config.epochs-1:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics
            }
            torch.save(checkpoint, os.path.join(Config.save_dir, f"checkpoint_epoch_{epoch+1}.pth"))
        
        print(f"Epoch {epoch+1}/{Config.epochs} | Loss: {avg_loss:.4f}")
        print_ct_metrics(val_metrics)
    
    # 保存最终模型和训练历史
    torch.save(model.state_dict(), os.path.join(Config.save_dir, "final_ct_model.pth"))
    save_training_history(history)
    visualize_ct_results(model, val_loader, Config.save_dir)
    
    return history

def evaluate_ct(model, data_loader, device):
    model.eval()
    metrics = SegmentationMetrics(Config.num_classes)
    dist_metrics = {'ASSD': [], 'HD': [], 'MSD': []}
    
    with torch.no_grad():
        for images, masks in tqdm(data_loader, desc="CT验证评估"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # 自动混合精度
            with autocast(enabled=Config.amp):
                outputs = model(images)
                preds = outputs.argmax(dim=1)
            
            # 更新分类指标
            metrics.update(preds.cpu().numpy().flatten(), masks.cpu().numpy().flatten())
            
            # 表面距离计算（器官级）
            for i in range(masks.size(0)):
                for organ_id in range(1, Config.num_classes):  # 跳过背景
                    gt_mask = (masks[i] == organ_id).cpu().byte().numpy()
                    pred_mask = (preds[i] == organ_id).cpu().byte().numpy()
                    
                    if gt_mask.sum() == 0 or pred_mask.sum() == 0:
                        continue
                    
                    surface_dists = compute_all_surface_metrics(gt_mask, pred_mask)
                    dist_metrics['ASSD'].append(surface_dists['ASSD'])
                    dist_metrics['HD'].append(surface_dists['HD'])
                    dist_metrics['MSD'].append(surface_dists['MSD'])
    
    # 合并指标
    class_metrics = metrics.compute()
    result = {
        'mIoU': class_metrics['mIoU'],
        'mPA': class_metrics['mPA'],
        'mPrecision': class_metrics['mPrecision'],
        'mRecall': class_metrics['mRecall'],
        'Dice': class_metrics['mDice'],
        'ASSD': np.mean(dist_metrics['ASSD']) if dist_metrics['ASSD'] else 0,
        'HD': np.mean(dist_metrics['HD']) if dist_metrics['HD'] else 0,
        'MSD': np.mean(dist_metrics['MSD']) if dist_metrics['MSD'] else 0
    }
    
    return result

def compute_surface_distances(mask_gt, mask_pred, spacing=(1.0, 1.0)):
    """CT图像专用表面距离计算"""
    if mask_gt.sum() == 0 or mask_pred.sum() == 0:
        return 0, 0
    
    # 精确边界计算
    gt_border = ndimage.binary_erosion(mask_gt) ^ mask_gt
    pred_border = ndimage.binary_erosion(mask_pred) ^ mask_pred
    
    # 距离变换
    gt_dist = ndimage.distance_transform_edt(~gt_border, sampling=spacing)
    pred_dist = ndimage.distance_transform_edt(~pred_border, sampling=spacing)
    
    # 距离提取
    dist_gt_to_pred = gt_dist[pred_border]
    dist_pred_to_gt = pred_dist[gt_border]
    
    return dist_gt_to_pred, dist_pred_to_gt

def compute_all_surface_metrics(mask_gt, mask_pred):
    """计算所有表面距离指标"""
    # 只计算有目标的区域
    if mask_gt.sum() == 0 or mask_pred.sum() == 0:
        return {'ASSD': 0, 'HD': 0, 'MSD': 0}
    
    # 计算表面距离
    dist_gt_to_pred, dist_pred_to_gt = compute_surface_distances(mask_gt, mask_pred)
    
    # 排除无效值
    dist_gt_to_pred = dist_gt_to_pred[~np.isinf(dist_gt_to_pred)]
    dist_pred_to_gt = dist_pred_to_gt[~np.isinf(dist_pred_to_gt)]
    
    if len(dist_gt_to_pred) == 0 or len(dist_pred_to_gt) == 0:
        return {'ASSD': 0, 'HD': 0, 'MSD': 0}
    
    # 计算指标
    assd = 0.5 * (dist_gt_to_pred.mean() + dist_pred_to_gt.mean())
    hd = max(dist_gt_to_pred.max(), dist_pred_to_gt.max())
    
    return {
        'ASSD': assd,
        'HD': hd,
        'MSD': assd  # MSD通常与ASSD相同
    }

def print_ct_metrics(metrics):
    """CT专用指标打印"""
    print(f"mIoU: {metrics['mIoU']:.4f} | mPA: {metrics['mPA']:.4f}")
    print(f"mPrecision: {metrics['mPrecision']:.4f} | mRecall: {metrics['mRecall']:.4f}")
    print(f"Dice: {metrics['Dice']:.4f}")
    print(f"MSD: {metrics['MSD']:.2f}mm | HD: {metrics['HD']:.2f}mm | ASSD: {metrics['ASSD']:.2f}mm")

def save_training_history(history):
    """保存CT训练历史"""
    os.makedirs(Config.save_dir, exist_ok=True)
    pd.DataFrame(history).to_csv(os.path.join(Config.save_dir, 'ct_metrics.csv'), index=False)
    
    # 绘制指标曲线
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(['mIoU', 'mPA', 'Dice', 'ASSD', 'HD', 'MSD'], 1):
        plt.subplot(2, 3, i)
        if metric in history:
            plt.plot(history['Epoch'], history[metric], 'o-')
            plt.title(metric)
            plt.xlabel('Epoch')
            plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.save_dir, 'ct_training_metrics.png'))
    plt.close()

def visualize_ct_results(model, data_loader, save_dir, num_samples=5):
    """CT分割结果可视化"""
    model.eval()
    samples = []
    indices = random.sample(range(len(data_loader.dataset)), num_samples)
    for idx in indices:
        samples.append(data_loader.dataset[idx])
    
    plt.figure(figsize=(18, 4 * num_samples))
    
    for i, (img, mask) in enumerate(samples):
        # 原始CT图像（反向归一化）
        img = img[:1]  # 取单通道
        img = img * Config.normalize_std[0] + Config.normalize_mean[0]
        ct_img = img[0].numpy()
        
        # 预测
        with torch.no_grad():
            output = model(img.unsqueeze(0).to(Config.device))
            pred = output.argmax(dim=1)[0].cpu().numpy()
        
        # 可视化
        plt.subplot(num_samples, 4, i*4+1)
        plt.imshow(ct_img, cmap='gray')
        plt.title("CT原图")
        
        plt.subplot(num_samples, 4, i*4+2)
        plt.imshow(mask.numpy(), cmap='jet')
        plt.title("真实分割")
        
        plt.subplot(num_samples, 4, i*4+3)
        plt.imshow(pred, cmap='jet')
        plt.title("预测分割")
        
        plt.subplot(num_samples, 4, i*4+4)
        plt.imshow(ct_img, cmap='gray')
        plt.imshow(pred, alpha=0.3, cmap='jet')
        plt.title("叠加效果")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ct_seg_results.png'))
    plt.close()

# ====================== 分割指标计算 ======================
class SegmentationMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.conf_matrix = np.zeros((num_classes, num_classes))
    
    def update(self, pred, target):
        """更新混淆矩阵"""
        mask = (target >= 0) & (target < self.num_classes)
        hist = np.bincount(
            self.num_classes * target[mask].astype(int) + pred[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        self.conf_matrix += hist
    
    def compute(self):
        """计算所有指标"""
        metrics = {}
        tp = np.diag(self.conf_matrix)
        fp = self.conf_matrix.sum(axis=0) - tp
        fn = self.conf_matrix.sum(axis=1) - tp
        
        # IoU
        iou = tp / (tp + fp + fn + 1e-10)
        metrics['IoU'] = iou
        metrics['mIoU'] = np.nanmean(iou)
        
        # 准确率
        accuracy = (tp) / (tp + fp + 1e-10)
        metrics['PA'] = accuracy
        metrics['mPA'] = np.nanmean(accuracy)
        
        # 精确率
        precision = tp / (tp + fp + 1e-10)
        metrics['Precision'] = precision
        metrics['mPrecision'] = np.nanmean(precision)
        
        # 召回率
        recall = tp / (tp + fn + 1e-10)
        metrics['Recall'] = recall
        metrics['mRecall'] = np.nanmean(recall)
        
        # Dice系数
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-10)
        metrics['Dice'] = dice
        metrics['mDice'] = np.nanmean(dice)
        
        return metrics
    
    def reset(self):
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes))

# ====================== 主程序 ======================
if __name__ == "__main__":
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    
    # 训练模型
    start_time = time.time()
    history = train_ct_model()
    end_time = time.time()
    
    # 输出最终结果
    final_metrics = {k: v[-1] if k != 'OrganAreas' else v for k, v in history.items() 
                    if k in ['mIoU', 'mPA', 'mPrecision', 'mRecall', 'Dice', 'MSD', 'HD', 'ASSD', 'OrganAreas']}
    
    print("\n===== 最终CT分割指标 =====")
    print_ct_metrics(final_metrics)
    
    # 保存统计信息
    stats = {}
    for metric in ['mIoU', 'mPA', 'mPrecision', 'mRecall', 'Dice', 'MSD', 'HD', 'ASSD']:
        if metric in history:
            values = history[metric]
            stats[f'{metric}_mean'] = np.mean(values)
            stats[f'{metric}_std'] = np.std(values)
    
    # 添加器官面积统计
    if 'OrganAreas' in history:
        for org in history['OrganAreas']:
            label = org['mapped_label']
            stats[f'organ_{label}_area'] = org['pixel_count']
            stats[f'organ_{label}_percent'] = org['relative_percentage']
    
    with open(os.path.join(Config.save_dir, 'ct_metrics_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"总训练时间: {(end_time-start_time)/3600:.2f}小时")
    print(f"结果保存至: {Config.save_dir}")