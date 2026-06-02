import torch


def calculate_metrics(preds, labels, poison_class_idx=1):
    """
    计算准确率和指定类别的召回率
    preds: 模型输出的 logits
    labels: 真实标签
    """
    # 经过 Sigmoid 后，>0.5 视为正类 (1)
    probs = torch.sigmoid(preds)
    predictions = (probs > 0.5).int()

    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total

    # 计算毒蘑菇(假设标签为1)的召回率: True Positives / (True Positives + False Negatives)
    true_positives = ((predictions == poison_class_idx) & (labels == poison_class_idx)).sum().item()
    actual_positives = (labels == poison_class_idx).sum().item()

    recall = true_positives / actual_positives if actual_positives > 0 else 0.0

    return accuracy, recall