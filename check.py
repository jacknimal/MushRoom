import os
from PIL import Image


def check_images(data_dir):
    bad_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, file)
                try:
                    img = Image.open(path)
                    img.load()  # 强行加载图像数据，这会触发截断错误
                except Exception as e:
                    print(f"❌ 损坏的文件: {path} | 错误: {e}")
                    bad_files.append(path)

    if bad_files:
        print(f"\n发现 {len(bad_files)} 个损坏的文件。建议直接删除。")
        # 如果你想自动删除，可以取消下面这一行的注释：
        # for f in bad_files: os.remove(f)
    else:
        print("\n✅ 所有图片均正常！")


# 填入你自己的数据集路径
check_images("/home/y/PycharmProjects/MushRoom/dataset")