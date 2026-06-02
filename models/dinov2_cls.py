import torch
import torch.nn as nn
import torch.nn.functional as F


class PoisonMushroomClassifier(nn.Module):
    # 【修改】默认 num_classes 改为 2，以适配交叉熵损失
    def __init__(self, local_weight_path, hidden_dim=256, num_classes=2):
        super(PoisonMushroomClassifier, self).__init__()

        # 1. 从本地目录加载 DINOv2-Base
        self.backbone = torch.hub.load('models/backbones/facebookresearch_dinov2_main/dinov2',
                                       'dinov2_vitb14',
                                       source='local',
                                       pretrained=False)

        # 2. 加载本地权重
        state_dict = torch.load(local_weight_path, map_location="cpu")
        self.backbone.load_state_dict(state_dict)

        # 3. 获取特征维度 (DINOv2-B 是 768)
        embed_dim = self.backbone.embed_dim

        # 4. 【新增】构建投影头 (Projection Head)，专门用于对比学习
        # 将高维特征降维到 128 维，以便在特征空间中进行推拉
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128)
        )

        # 5. 构建分类头 (MLP)
        # 结构保持与你原来完全一致，只是最后一层输出变成了 2 (num_classes)
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # 提取 DINOv2 的特征
        features = self.backbone(x)

        # 1. 计算对比学习所需的 Embedding，并进行 L2 归一化 (InfoNCE 的数学要求)
        embeddings = self.proj_head(features)
        embeddings = F.normalize(embeddings, dim=1)

        # 2. 计算分类所需的 Logits
        logits = self.cls_head(features)

        # 【修改】同时返回 embeddings 和 logits
        return embeddings, logits