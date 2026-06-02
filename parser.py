import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="🍄 工业级 DINOv2 毒蘑菇二分类系统")

    # ==========================
    # 实验设置 (Experiment Setup)
    # ==========================
    parser.add_argument('--seed', type=int, default=0, help='全局随机种子')
    parser.add_argument('--img_size', type=int, default=224, help='输入图像大小 (必须是14的倍数)')

    # ==========================
    # 数据集与路径配置
    # ==========================
    parser.add_argument('--data_dir', type=str, default='/home/y/PycharmProjects/MushRoom/dataset', help='数据集根目录')
    parser.add_argument('--save_dir', type=str, default='./Result/out', help='最佳模型权重保存目录')
    parser.add_argument('--base_weight_path', type=str, default='./weights/dinov2_vitb14_pretrain.pth',
                        help='DINOv2-B 本地权重路径')

    # ==========================
    # 硬件与数据加载
    # ==========================
    parser.add_argument('--gpu_id', type=str, default='0', help='指定使用的 GPU 编号 (例如: 0, 1 或 0,1)')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=8, help='加载数据线程数')

    # ==========================
    # 核心：网络结构进阶设计
    # ==========================
    parser.add_argument('--head_hidden_dim', type=int, default=256,
                        help='分类头 MLP 的隐藏层维度 (建议值: 256, 512)')

    # ==========================
    # 训练与优化器超参数
    # ==========================
    parser.add_argument('--epochs', type=int, default=10, help='总训练轮数')
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help='骨干网络微调学习率')
    parser.add_argument('--lr_head', type=float, default=1e-3, help='分类头学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='AdamW 权重衰减')

    # 新增：学习率调度器参数
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='余弦退火的最小学习率')

    # 代价敏感学习配置
    parser.add_argument('--pos_weight', type=float, default=1.0, help='毒蘑菇类的 BCE 惩罚权重')

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.img_size % 14 != 0:
        print(f"⚠️ 警告: img_size ({args.img_size}) 不是 14 的倍数！")

    return args