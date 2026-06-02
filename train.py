import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

from data.dataset import get_dataloaders
from models.dinov2_cls import PoisonMushroomClassifier
from parser import parse_args
from utils.seed import set_seed


# ==============================================================
# 【新增】标准的有监督对比学习损失函数 (Supervised Contrastive Loss)
# 直接内置在 train.py 中，无需额外引包
# ==============================================================
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')

        batch_size = features.shape[0]
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)  # 加上 1e-9 防止 log(0)

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


# ==============================================================


def main():
    args = parse_args()
    set_seed(args.seed)

    # 【新增】设置系统环境变量以屏蔽不需要的显卡
    # 这一行必须放在 torch.device 定义之前！
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # 因为上面已经屏蔽了其他显卡，这里的 "cuda" 会自动映射到你指定的 args.gpu_id 上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )

    model = PoisonMushroomClassifier(
        local_weight_path=args.base_weight_path,
        hidden_dim=args.head_hidden_dim
    ).to(device)

    # 1. 精确匹配每一层的参数名字，防止重叠
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
    cls_head_params = [p for n, p in model.named_parameters() if "cls_head" in n]
    proj_head_params = [p for n, p in model.named_parameters() if "proj_head" in n]

    # 2. 将它们分组送入优化器
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr_backbone},
        {'params': cls_head_params, 'lr': args.lr_head},
        {'params': proj_head_params, 'lr': args.lr_head}  # 投影头和分类头用同样的学习率
    ], weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr
    )

    # 【修改】损失函数改为 交叉熵 + 有监督对比学习
    criterion_ce = nn.CrossEntropyLoss()
    criterion_supcon = SupConLoss(temperature=0.07).to(device)
    alpha = 0.5  # 对比学习损失的权重系数 (0.5是经典经验值)

    best_recall = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        current_head_lr = optimizer.param_groups[1]['lr']
        print(f"\n--- Epoch [{epoch + 1}/{args.epochs}] | Head LR: {current_head_lr:.6f} ---")

        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{args.epochs}")
        for inputs, labels in train_pbar:
            # 【修改】交叉熵要求 labels 是 1维的 LongTensor
            inputs, labels = inputs.to(device), labels.to(device).long()

            optimizer.zero_grad()

            # 【修改】接收两个输出
            embeddings, logits = model(inputs)

            # 计算交叉熵分类损失
            loss_ce = criterion_ce(logits, labels)
            # 计算对比学习损失 (SupCon 需要加一个 view 维度，所以用 unsqueeze(1))
            loss_supcon = criterion_supcon(embeddings.unsqueeze(1), labels)

            # 联合损失
            loss = loss_ce + alpha * loss_supcon

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()

        print(f"Train Loss: {train_loss / len(train_loader.dataset):.4f}")

        # --- 验证循环 ---
        model.eval()
        val_loss = 0.0

        TP = 0
        TN = 0
        FP = 0
        FN = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Valid Epoch {epoch + 1}/{args.epochs}")
            for inputs, labels in val_pbar:
                # 【修改】保持数据类型和验证期间的一致性
                inputs, labels = inputs.to(device), labels.to(device).long()

                embeddings, logits = model(inputs)

                # 验证集也计算一下总 Loss 观察是否过拟合
                loss_ce = criterion_ce(logits, labels)
                loss_supcon = criterion_supcon(embeddings.unsqueeze(1), labels)
                loss = loss_ce + alpha * loss_supcon
                val_loss += loss.item() * inputs.size(0)

                # 【修改】使用 argmax 获取类别索引 (0: 无毒, 1: 有毒)，替代原来的 sigmoid
                predictions = torch.argmax(logits, dim=1)

                # 统计 TP, TN, FP, FN (逻辑保持不变)
                TP += ((predictions == 1) & (labels == 1)).sum().item()
                TN += ((predictions == 0) & (labels == 0)).sum().item()
                FP += ((predictions == 1) & (labels == 0)).sum().item()
                FN += ((predictions == 0) & (labels == 1)).sum().item()

        val_loss /= len(val_loader.dataset)
        total_samples = TP + TN + FP + FN
        val_acc = (TP + TN) / total_samples if total_samples > 0 else 0

        actual_poisonous = TP + FN
        val_poison_recall = TP / actual_poisonous if actual_poisonous > 0 else 0
        val_poison_precision = TP / (TP + FP) if (TP + FP) > 0 else 0

        actual_edible = TN + FP
        val_edible_recall = TN / actual_edible if actual_edible > 0 else 0
        val_edible_precision = TN / (TN + FN) if (TN + FN) > 0 else 0

        print(f"\n📊 --- Epoch [{epoch + 1}/{args.epochs}] 验证集详细报告 ---")
        print(f"📉 Val Loss: {val_loss:.4f} | 🎯 整体准确率 (Acc): {val_acc * 100:.2f}%")

        print(f"☠️ [有毒类] 实际总数: {actual_poisonous} | 成功拦截(TP): {TP} | 致命漏判(FN): {FN}")
        print(
            f"   -> 判毒正确率 (Recall): {val_poison_recall * 100:.2f}% | 报毒精确率: {val_poison_precision * 100:.2f}%")

        print(f"🥗 [无毒类] 实际总数: {actual_edible} | 成功放行(TN): {TN} | 误杀(FP): {FP}")
        print(
            f"   -> 判无毒正确率 (Specificity): {val_edible_recall * 100:.2f}% | 报无毒精确率: {val_edible_precision * 100:.2f}%")
        print("-" * 60)

        # 保存最佳权重 (以毒蘑菇的 Recall 为最优先指标)
        if val_poison_recall > best_recall:
            best_recall = val_poison_recall
            save_path = os.path.join(args.save_dir, "best_poison_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f">>> 🍄 发现更强模型！已保存最佳权重 (Poison Recall: {best_recall * 100:.2f}%) 至 {save_path} <<<\n")
        else:
            print("\n")


if __name__ == '__main__':
    main()