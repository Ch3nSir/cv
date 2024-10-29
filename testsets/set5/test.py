from PIL import Image
import os

# 遍历指定路径下的所有图片文件
directory_path = './'  # 替换为你的目录路径
for filename in os.listdir(directory_path):
    if filename.endswith('.bmp') or filename.endswith('.png') or filename.endswith('.jpg'):
        image_path = os.path.join(directory_path, filename)
        
        # 打开图片
        image = Image.open(image_path)
        
        # 获取图像的模式
        mode = image.mode
        
        # 输出通道数量
        channel_count = len(mode)
        print(f'Image: {filename}, Mode: {mode}, Channel count: {channel_count}')


