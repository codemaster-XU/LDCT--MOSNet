import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------------- 数据读取 ----------------
class CTMask2DDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_list):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        img = nib.load(os.path.join(self.image_dir, fname + ".nii.gz")).get_fdata().astype(np.float32)
        mask = nib.load(os.path.join(self.mask_dir, fname + ".nii.gz")).get_fdata().astype(np.int64)
        # 只取第一帧（假如是3D），如本来就是2D则无影响
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]
            mask = mask[..., 0]
        elif img.ndim == 3:
            img = img[:, :, 0]
            mask = mask[:, :, 0]
        # 归一化
        img = (img - img.mean()) / (img.std() + 1e-8)
        # [C, H, W]，单通道
        img = np.expand_dims(img, 0)
        return torch.from_numpy(img), torch.from_numpy(mask)

# ---------------- 简单2D U-Net ----------------
class Simple2DUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, features=32):
        super().__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, padding=1), nn.ReLU(),
            nn.Conv2d(features, features, 3, padding=1), nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = nn.Sequential(
            nn.Conv2d(features, features*2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(features*2, features*2, 3, padding=1), nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*2, features*4, 3, padding=1), nn.ReLU(),
            nn.Conv2d(features*4, features*4, 3, padding=1), nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(features*4, features*2, 2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(features*4, features*2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(features*2, features*2, 3, padding=1), nn.ReLU()
        )
        self.up1 = nn.ConvTranspose2d(features*2, features, 2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(features*2, features, 3, padding=1), nn.ReLU(),
            nn.Conv2d(features, features, 3, padding=1), nn.ReLU()
        )
        self.out_conv = nn.Conv2d(features, out_channels, 1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        dec2 = self.up2(bottleneck)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        return self.out_conv(dec1)

# ---------------- 评价指标 ----------------
def compute_metrics(pred, gt, num_classes):
    ious, precisions, recalls, pas = [], [], [], []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        gt_cls = (gt == cls)
        intersection = (pred_cls & gt_cls).sum()
        union = (pred_cls | gt_cls).sum()
        tp = intersection
        fp = pred_cls.sum() - tp
        fn = gt_cls.sum() - tp
        tn = (~(pred_cls | gt_cls)).sum()
        iou = intersection / (union + 1e-8)
        pa = (tp + tn) / (tp + fp + fn + tn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        ious.append(iou)
        precisions.append(precision)
        recalls.append(recall)
        pas.append(pa)
    mIOU = np.mean(ious)
    mPA = np.mean(pas)
    mPrecision = np.mean(precisions)
    mRecall = np.mean(recalls)
    return {
        "mIOU": mIOU,
        "mPA": mPA,
        "mPrecision": mPrecision,
        "mRecall": mRecall,
        "ious": ious,
        "precisions": precisions,
        "recalls": recalls,
        "pas": pas
    }

# ---------------- 主流程 ----------------
def main():
    # 路径配置
    image_dir = r"D:\your_ct_folder"      # 替换为你的CT原图文件夹路径
    mask_dir = r"D:\your_mask_folder"     # 替换为你的mask标签文件夹路径
    all_files = [os.path.splitext(os.path.splitext(f)[0])[0] for f in os.listdir(image_dir) if f.endswith(".nii.gz")]
    np.random.shuffle(all_files)
    split = int(len(all_files) * 0.8)
    train_files = all_files[:split]
    val_files = all_files[split:]

    # 数据集
    train_ds = CTMask2DDataset(image_dir, mask_dir, train_files)
    val_ds = CTMask2DDataset(image_dir, mask_dir, val_files)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Simple2DUNet(in_channels=1, out_channels=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 训练
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        print(f"Epoch {epoch+1} finished, avg loss: {epoch_loss/len(train_loader):.4f}")

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # 验证
    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Validating"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_gts.append(masks.numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)
    metrics = compute_metrics(all_preds, all_gts, num_classes=2)
    print("mIOU:", metrics["mIOU"])
    print("mPA:", metrics["mPA"])
    print("mPrecision:", metrics["mPrecision"])
    print("mRecall:", metrics["mRecall"])

if __name__ == "__main__":
    main()