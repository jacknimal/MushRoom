import os
import shutil
import random


def split_and_copy(src_dir, train_dir, val_dir, train_num, val_num):
    """
    从源文件夹随机选取图片，互斥地划分到训练集和验证集并复制。
    """
    print(f"========================================")
    print(f"正在处理来源: {src_dir}")

    # 1. 确保目标文件夹存在，如果不存在则自动创建
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 2. 获取源目录下的所有文件（过滤掉可能存在的子文件夹）
    all_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    total_files = len(all_files)
    print(f"共发现 {total_files} 张图片。")

    # 3. 检查源文件夹中的图片数量是否足够
    total_needed = train_num + val_num
    if total_files < total_needed:
        raise ValueError(f"❌ 错误：{src_dir} 中的图片数量不足！需要 {total_needed} 张，但只有 {total_files} 张。")

    # 4. 设定随机种子并打乱文件列表
    # 设定种子可以保证如果你不小心删了重跑，切分的结果是一模一样的
    random.seed(42)
    random.shuffle(all_files)

    # 5. 通过切片划分文件，确保 train 和 val 绝对不会有交集
    train_files = all_files[:train_num]
    val_files = all_files[train_num: train_num + val_num]

    # 6. 复制文件到 train 文件夹
    print(f"\n📂 开始复制 {train_num} 张图片到 {train_dir} ...")
    for i, f in enumerate(train_files):
        src_path = os.path.join(src_dir, f)
        dst_path = os.path.join(train_dir, f)
        shutil.copy(src_path, dst_path)
        # 每复制 5000 张打印一次进度，避免满屏输出
        if (i + 1) % 5000 == 0 or (i + 1) == train_num:
            print(f"   -> 已复制 {i + 1} / {train_num}")

    # 7. 复制文件到 val 文件夹
    print(f"\n📂 开始复制 {val_num} 张图片到 {val_dir} ...")
    for i, f in enumerate(val_files):
        src_path = os.path.join(src_dir, f)
        dst_path = os.path.join(val_dir, f)
        shutil.copy(src_path, dst_path)
        if (i + 1) % 1000 == 0 or (i + 1) == val_num:
            print(f"   -> 已复制 {i + 1} / {val_num}")

    print(f"✅ {src_dir} 处理完成！\n")


if __name__ == '__main__':
    # ==========================
    # 任务 1: 处理有毒蘑菇 (youdu)
    # ==========================
    src_poisonous = "/home/y/strong_labels/poisonous"
    train_poisonous = "/home/y/PycharmProjects/MushRoom/dataset/train/youdu"
    val_poisonous = "/home/y/PycharmProjects/MushRoom/dataset/val/youdu"

    # 需求: 22984 进 train, 2554 进 val
    split_and_copy(src_poisonous, train_poisonous, val_poisonous, train_num=22984, val_num=2554)

    # ==========================
    # 任务 2: 处理无毒蘑菇 (wudu)
    # ==========================
    src_edible = "/home/y/strong_labels/edible"
    train_edible = "/home/y/PycharmProjects/MushRoom/dataset/train/wudu"
    val_edible = "/home/y/PycharmProjects/MushRoom/dataset/val/wudu"

    # 需求: 22984 进 train, 2354 进 val
    split_and_copy(src_edible, train_edible, val_edible, train_num=22984, val_num=2554)

    print("🎉 所有数据划分和复制任务已圆满完成！你可以开始训练了！")