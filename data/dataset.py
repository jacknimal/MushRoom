import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_transforms(img_size=224):  # <--- 增加 img_size 参数
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),  # 使用 img_size
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),  # 通常 Resize 到比 Crop 大一点点 (256/224 ≈ 1.14)
        transforms.CenterCrop(img_size),  # 使用 img_size
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    return train_transform, val_transform


def get_dataloaders(data_dir, batch_size=64, num_workers=8, img_size=224):
    """
    假设你的 data_dir 下有 'train' 和 'val' 两个文件夹
    每个文件夹下有 'poisonous' 和 'edible' 两个子文件夹
    """
    train_transform, val_transform = get_transforms(img_size)

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

    # 打印类别映射，确保我们知道哪个类标是毒蘑菇 (通常索引 1)
    print(f"类别映射: {train_dataset.class_to_idx}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader