import torch
import os
from data.dataset import get_dataloaders
from models.dinov2_cls import PoisonMushroomClassifier


def evaluate_model():
    # --- 路径与参数配置 (请根据你最新的 parser.py 参数对应修改) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "/home/y/PycharmProjects/MushRoom/dataset"  # 测试集/验证集所在的根目录
    model_weight_path = "./Result/out/best_poison_model.pth"  # 你训练好的最优权重
    base_dinov2_weight = "./weights/dinov2_vitb14_pretrain.pth"
    batch_size = 64
    img_size = 224

    print("⏳ 正在加载数据和模型...")
    # 仅需要验证/测试数据，忽略训练集
    _, test_loader = get_dataloaders(data_dir, batch_size=batch_size, img_size=img_size)

    # 实例化并加载你训练好的权重
    # 注意：如果你之前采纳了建议，加上了 hidden_dim=256，这里需要传同样的参数
    model = PoisonMushroomClassifier(base_dinov2_weight, hidden_dim=256).to(device)

    if os.path.exists(model_weight_path):
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        print(f"✅ 成功加载权重: {model_weight_path}")
    else:
        print(f"❌ 找不到权重文件: {model_weight_path}")
        return

    model.eval()

    # ==========================
    # 初始化混淆矩阵变量
    # ==========================
    TP = 0  # True Positive:  实际有毒，预测有毒 (成功逮住毒蘑菇)
    TN = 0  # True Negative:  实际无毒，预测无毒 (成功认出好蘑菇)
    FP = 0  # False Positive: 实际无毒，预测有毒 (虚惊一场，把好蘑菇当毒蘑菇扔了)
    FN = 0  # False Negative: 实际有毒，预测无毒 (致命错误！毒蘑菇上了餐桌)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)

            # 经过 Sigmoid 后，>0.5 视为有毒 (1)，<=0.5 视为无毒 (0)
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).int()

            # 统计 TP, TN, FP, FN
            TP += ((predictions == 1) & (labels == 1)).sum().item()
            TN += ((predictions == 0) & (labels == 0)).sum().item()
            FP += ((predictions == 1) & (labels == 0)).sum().item()
            FN += ((predictions == 0) & (labels == 1)).sum().item()

    # ==========================
    # 计算详细指标
    # ==========================
    total_samples = TP + TN + FP + FN
    overall_acc = (TP + TN) / total_samples if total_samples > 0 else 0

    # 类别 1: 有毒蘑菇的指标
    actual_poisonous = TP + FN
    poison_recall = TP / actual_poisonous if actual_poisonous > 0 else 0  # 判毒正确率
    poison_precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # 说它有毒，真的有毒的概率

    # 类别 0: 无毒蘑菇的指标
    actual_edible = TN + FP
    edible_recall = TN / actual_edible if actual_edible > 0 else 0  # 判无毒正确率
    edible_precision = TN / (TN + FN) if (TN + FN) > 0 else 0  # 说它无毒，真的无毒的概率 (吃下去的安全概率)

    # ==========================
    # 打印超级详细的测试报告
    # ==========================
    print("\n" + "=" * 50)
    print(" 🍄 毒蘑菇二分类系统 —— 深度评估报告 🍄 ")
    print("=" * 50)
    print(f"总测试样本数: {total_samples}")
    print(f"整体准确率 (Overall Accuracy): {overall_acc * 100:.2f}%\n")

    print("☠️  【有毒蘑菇】(类别 1) 指标:")
    print(f"   -> 测试集中实际有毒总数: {actual_poisonous}")
    print(f"   -> 成功拦截 (TP): {TP}")
    print(f"   -> 致命漏判 (FN): {FN}  <-- 重点关注，越低越好！")
    print(f"   -> 判断有毒的正确率 (Recall):      {poison_recall * 100:.2f}%")
    print(f"   -> 报毒的精确率 (Precision):       {poison_precision * 100:.2f}%\n")

    print("🥗 【无毒蘑菇】(类别 0) 指标:")
    print(f"   -> 测试集中实际无毒总数: {actual_edible}")
    print(f"   -> 成功放行 (TN): {TN}")
    print(f"   -> 误杀丢弃 (FP): {FP}  <-- 虚惊一场")
    print(f"   -> 判断无毒的正确率 (Specificity): {edible_recall * 100:.2f}%")
    print(f"   -> 报无毒的精确率 (Precision):       {edible_precision * 100:.2f}%  <-- 极度关键：它说无毒，你敢不敢吃？")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    evaluate_model()