# 欢迎来到全新的 nnU-Net！

如果你在寻找旧版本，请点击[这里](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1)。

从 V1 迁移过来？请查看[简明迁移指南](documentation/tldr_migration_guide_from_v1.md)。强烈建议你继续阅读其余文档 ;-)

## **2024-04-18 更新：全新残差编码器 UNet 预设已发布！**
残差编码器 UNet 预设大幅提升了分割性能。
它们支持多种 GPU 显存目标。真的很棒，保证你会喜欢！ 
详细内容请阅读 :point_right: [这里](documentation/resenc_presets.md) :point_left:

还请查阅我们的[新论文](https://arxiv.org/pdf/2404.09556.pdf)，系统性地对医学图像分割领域的最新进展进行了基准测试。你可能会感到惊讶！

# 什么是 nnU-Net？
图像数据集极其多样：图像维度（2D、3D）、模态/输入通道（RGB 图像、CT、MRI、显微镜等）、图像大小、体素大小、类别比例、目标结构属性等在不同数据集之间变化巨大。
传统上，面对新问题时，需要手动设计和优化定制化的解决方案——这个过程容易出错、难以扩展，且成功与否极大依赖于实验者的能力。即使是专家，这个过程也绝非易事：不仅有许多设计选择和数据属性需要考虑，而且它们彼此紧密关联，使得可靠的手动流程优化几乎不可能！

![nnU-Net overview](documentation/assets/nnU-Net_overview.png)

**nnU-Net 是一种能够自动适应给定数据集的语义分割方法。它会分析你提供的训练样本，并自动配置匹配的基于 U-Net 的分割流程。你无需任何专业知识！你只需训练模型并将其用于你的应用即可。**

在发布时，nnU-Net 在来自生物医学领域竞赛的 23 个数据集上进行了评估。尽管与每个数据集的手工定制方案竞争，nnU-Net 的全自动流程在多个公开排行榜上获得了第一名！自那以后，nnU-Net 经受住了时间的考验：它持续被用作基线和方法开发框架（[MICCAI 2020 年 10 个挑战赛冠军中有 9 个](https://arxiv.org/abs/2101.00232)和 2021 年 7 个中有 5 个都基于 nnU-Net，[我们用 nnU-Net 赢得了 AMOS2022](https://amos22.grand-challenge.org/final-ranking/)）！

如果你使用 nnU-Net，请引用[以下论文](https://www.nature.com/articles/s41592-020-01008-z)：

    Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
    method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.

## nnU-Net 能为你做什么？
如果你是**领域科学家**（生物学家、放射科医生等），希望分析自己的图像，nnU-Net 提供了开箱即用的解决方案，几乎可以保证在你的数据集上获得优异结果。只需将你的数据集转换为 nnU-Net 格式，尽享 AI 的强大——无需专业知识！

如果你是**AI 研究者**，正在开发分割方法，nnU-Net：
- 提供了极佳的开箱即用基线算法，便于对比
- 可作为方法开发框架，在大量数据集上测试你的创新，无需为每个流程单独调优（例如评估新的损失函数）
- 为进一步针对数据集优化提供了强有力的起点，尤其适用于分割挑战赛
- 为分割方法设计提供了新视角：也许你能发现数据集属性与最佳分割流程之间更好的联系？

## nnU-Net 的适用范围是什么？
nnU-Net 专为语义分割而设计。它可以处理具有任意输入模态/通道的 2D 和 3D 图像。它能理解体素间距、各向异性，即使类别极度不平衡也很稳健。

nnU-Net 依赖于监督学习，这意味着你需要为你的应用提供训练样本。所需训练样本数量取决于分割问题的复杂性，没有统一标准！nnU-Net 并不比其他方案需要更多训练样本——由于我们大量使用数据增强，甚至可能更少。

nnU-Net 期望在预处理和后处理阶段能一次性处理整张图像，因此无法处理超大图像。作为参考：我们测试过 3D 图像从 40x40x40 到 1500x1500x1500，2D 图像从 40x40 到约 30000x30000！如果你的内存允许，更大也没问题。

## nnU-Net 如何工作？
面对新数据集，nnU-Net 会系统性地分析提供的训练样本并创建“数据集指纹”。
然后，nnU-Net 为每个数据集创建多个 U-Net 配置：
- `2d`：2D U-Net（适用于 2D 和 3D 数据集）
- `3d_fullres`：在高分辨率图像上运行的 3D U-Net（仅适用于 3D 数据集）
- `3d_lowres` → `3d_cascade_fullres`：3D U-Net 级联，先在低分辨率图像上运行 3D U-Net，再由第二个高分辨率 3D U-Net 精细化前者的预测（仅适用于大尺寸 3D 数据集）

**注意：并非所有数据集都会创建所有 U-Net 配置。对于小尺寸图像数据集，会省略 U-Net 级联（以及 3d_lowres 配置），因为全分辨率 U-Net 的 patch 已覆盖了大部分输入图像。**

nnU-Net 的分割流程基于三步法配置：
- **固定参数**不会被调整。在 nnU-Net 开发过程中，我们确定了一套稳健的配置（即某些架构和训练属性），可始终使用。例如 nnU-Net 的损失函数、大部分数据增强策略和学习率。
- **基于规则的参数**利用数据集指纹，通过硬编码的启发式规则调整某些分割流程属性。例如，网络结构（池化行为和深度）会根据 patch 大小调整；patch 大小、网络结构和 batch size 会在 GPU 显存约束下联合优化。
- **经验参数**本质上是试错。例如为给定数据集选择最佳 U-Net 配置（2D、3D 全分辨率、3D 低分辨率、3D 级联）以及优化后处理策略。

## 如何开始？
请阅读以下内容：
- [安装说明](documentation/installation_instructions.md)
- [数据集转换](documentation/dataset_format.md)
- [使用说明](documentation/how_to_use_nnunet.md)

附加信息：
- [从稀疏标注（涂鸦、切片）中学习](documentation/ignore_label.md)
- [基于区域的训练](documentation/region_based_training.md)
- [手动数据划分](documentation/manual_data_splits.md)
- [预训练和微调](documentation/pretraining_and_finetuning.md)
- [nnU-Net中的强度归一化](documentation/explanation_normalization.md)
- [手动编辑nnU-Net配置](documentation/explanation_plans_files.md)
- [扩展nnU-Net](documentation/extending_nnunet.md)
- [V2中有什么不同？](documentation/changelog.md)

竞赛相关：
- [AutoPET II](documentation/competitions/AutoPETII.md)

[//]: # (- [Ignore label]&#40;documentation/ignore_label.md&#41;)

## nnU-Net 擅长哪些场景，不适合哪些场景？
nnU-Net 在需要从零训练的分割问题中表现出色，
例如：具有非标准图像模态和输入通道的科研应用、生物医学领域的挑战赛数据集、大多数 3D 分割问题等。我们尚未发现 nnU-Net 工作原理失效的数据集！

注意：在标准分割问题（如 ADE20k 和 Cityscapes 的 2D RGB 图像）上，微调一个在大规模类似图像（如 Imagenet 22k、JFT-300M）上预训练的基础模型会比 nnU-Net 表现更好！这仅仅是因为这些模型能提供更好的初始化。nnU-Net 不支持基础模型，因为它们 1）对偏离标准设置的分割问题无用（见上文数据集），2）通常只支持 2D 架构，3）与我们为每个数据集精细调整网络结构的核心设计原则冲突（如果结构变了就无法迁移预训练权重！）

## 旧版 nnU-Net 怎么了？
旧版 nnU-Net 的核心是在 2018 年参加 Medical Segmentation Decathlon 挑战赛时匆忙拼凑的。因此，代码结构和质量并不理想。后来添加了许多功能，也不太符合 nnU-Net 的设计原则。整体上相当混乱，使用起来也很烦人。

nnU-Net V2 是一次彻底重构。属于“全部删除重来”的那种。所以一切都更好了（作者如是说，哈哈）。虽然分割性能[保持不变](https://docs.google.com/spreadsheets/d/13gqjIKEMPFPyMMMwA1EML57IyoBjfC3-QCTn4zRN_Mg/edit?usp=sharing)，但增加了许多很酷的新功能。
现在用它做开发框架和手动微调新数据集的配置也容易多了。重构的一个重要原因是 [Helmholtz Imaging](http://helmholtz-imaging.de) 的出现，促使我们将 nnU-Net 扩展到更多图像格式和领域。更多亮点请见[这里](documentation/changelog.md)。

# 鸣谢
<img src="documentation/assets/HI_Logo.png" height="100px" />

<img src="documentation/assets/dkfz_logo.png" height="100px" />

nnU-Net 由 [Helmholtz Imaging](http://helmholtz-imaging.de) 的应用计算机视觉实验室（ACVL）和
[德国癌症研究中心（DKFZ）医学图像计算部](https://www.dkfz.de/en/mic/index.php)开发和维护。
- [基于区域的训练](documentation/region_based_training.md)
- [手动数据划分](documentation/manual_data_splits.md)
- [预训练和微调](documentation/pretraining_and_finetuning.md)
- [nnU-Net中的强度归一化](documentation/explanation_normalization.md)
- [手动编辑nnU-Net配置](documentation/explanation_plans_files.md)
- [扩展nnU-Net](documentation/extending_nnunet.md)
- [V2中有什么不同？](documentation/changelog.md)

Competitions:
- [AutoPET II](documentation/competitions/AutoPETII.md)

[//]: # (- [Ignore label]&#40;documentation/ignore_label.md&#41;)

## Where does nnU-Net perform well and where does it not perform?
nnU-Net excels in segmentation problems that need to be solved by training from scratch, 
for example: research applications that feature non-standard image modalities and input channels,
challenge datasets from the biomedical domain, majority of 3D segmentation problems, etc . We have yet to find a 
dataset for which nnU-Net's working principle fails!

Note: On standard segmentation 
problems, such as 2D RGB images in ADE20k and Cityscapes, fine-tuning a foundation model (that was pretrained on a large corpus of 
similar images, e.g. Imagenet 22k, JFT-300M) will provide better performance than nnU-Net! That is simply because these 
models allow much better initialization. Foundation models are not supported by nnU-Net as 
they 1) are not useful for segmentation problems that deviate from the standard setting (see above mentioned 
datasets), 2) would typically only support 2D architectures and 3) conflict with our core design principle of carefully adapting 
the network topology for each dataset (if the topology is changed one can no longer transfer pretrained weights!) 

## What happened to the old nnU-Net?
The core of the old nnU-Net was hacked together in a short time period while participating in the Medical Segmentation 
Decathlon challenge in 2018. Consequently, code structure and quality were not the best. Many features 
were added later on and didn't quite fit into the nnU-Net design principles. Overall quite messy, really. And annoying to work with.

nnU-Net V2 is a complete overhaul. The "delete everything and start again" kind. So everything is better 
(in the author's opinion haha). While the segmentation performance [remains the same](https://docs.google.com/spreadsheets/d/13gqjIKEMPFPyMMMwA1EML57IyoBjfC3-QCTn4zRN_Mg/edit?usp=sharing), a lot of cool stuff has been added. 
It is now also much easier to use it as a development framework and to manually fine-tune its configuration to new 
datasets. A big driver for the reimplementation was also the emergence of [Helmholtz Imaging](http://helmholtz-imaging.de), 
prompting us to extend nnU-Net to more image formats and domains. Take a look [here](documentation/changelog.md) for some highlights.

# Acknowledgements
<img src="documentation/assets/HI_Logo.png" height="100px" />

<img src="documentation/assets/dkfz_logo.png" height="100px" />

nnU-Net is developed and maintained by the Applied Computer Vision Lab (ACVL) of [Helmholtz Imaging](http://helmholtz-imaging.de) 
and the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php) at the 
[German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html).
