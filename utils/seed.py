import random
import numpy as np
import torch
import os


def set_seed(seed=42):
    """
    固定所有的随机种子，确保实验可复现 (Reproducibility)
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果有多个GPU

    # cuDNN 确定性设置 (可能会稍微牺牲一点点点卷积速度，但保证完全可复现)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"🌱 已全局锁定随机种子: {seed}")