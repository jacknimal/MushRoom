import torch
import os
from PIL import Image, ImageFile
from torchvision import transforms
import argparse
from tqdm import tqdm

# 容忍轻微损坏的图片
ImageFile.LOAD_TRUNCATED_IMAGES = True

from models.dinov2_cls import PoisonMushroomClassifier


def predict_folder(args):
    if not os.path.isdir(args.folder):
        print(f"❌ 找不到文件夹: {args.folder}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 正在使用设备: {device} 进行推理...")

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(int(args.img_size * 1.14)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # 加载模型
    print("⏳ 正在加载模型权重...")
    model = PoisonMushroomClassifier(
        local_weight_path=args.base_weight_path,
        hidden_dim=args.head_hidden_dim
    ).to(device)

    try:
        model.load_state_dict(torch.load(args.model_weight, map_location=device))
        print("✅ 模型权重加载成功！\n")
    except Exception as e:
        print(f"❌ 加载权重失败: {e}")
        return

    model.eval()

    # 读取文件夹内容
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [f for f in os.listdir(args.folder) if f.lower().endswith(valid_ext)]

    if len(image_files) == 0:
        print(f"⚠️ 文件夹 [{args.folder}] 中没有图片！")
        return

    poisonous_count = 0
    edible_count = 0

    print(f"📂 开始鉴定 {len(image_files)} 张图片...\n")

    with torch.no_grad():
        # 添加进度条进行批量预测
        for img_name in tqdm(image_files, desc="Predicting"):
            img_path = os.path.join(args.folder, img_name)

            try:
                image = Image.open(img_path).convert('RGB')
            except Exception:
                continue

            input_tensor = transform(image).unsqueeze(0).to(device)
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()

            if prob > 0.5:
                poisonous_count += 1
                # 如果你想看每一个的输出，可以取消注释下面这行
                # print(f"☠️ 【有毒】: {img_name} ({prob * 100:.1f}%)")
            else:
                edible_count += 1
                # print(f"🥗 【无毒】: {img_name} ({(1 - prob) * 100:.1f}%)")

    # 总结报告
    print("\n" + "=" * 50)
    print(" 🍄 批量蘑菇照妖镜 —— 盲测总结报告 🍄 ")
    print("=" * 50)
    print(f"📁 目标目录: {args.folder}")
    print(f"📸 成功预测: {poisonous_count + edible_count} 张")
    print(f"☠️ 发现毒蘑菇: {poisonous_count} 张")
    print(f"🥗 发现无毒菇: {edible_count} 张")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="批量蘑菇图片毒性盲测工具")
    # 必须提供的待测文件夹路径
    parser.add_argument('--folder', type=str, required=True, help='待鉴定图片所在的文件夹路径')
    # 权重路径
    parser.add_argument('--model_weight', type=str, default='./Result/out/0_best_poison_model.pth',
                        help='你训练好的最优权重')
    parser.add_argument('--base_weight_path', type=str, default='./weights/dinov2_vitb14_pretrain.pth',
                        help='DINOv2基础权重')
    # 架构配置 (务必一致)
    parser.add_argument('--img_size', type=int, default=224, help='输入图像大小')
    parser.add_argument('--head_hidden_dim', type=int, default=256, help='MLP隐藏层维度')

    args = parser.parse_args()
    predict_folder(args)