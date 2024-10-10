import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from evalPIQAtest import predict_mos
from HE_Based.illumination_boost import illumination_boost
import glob
import os

class MappingNN(nn.Module):
    def __init__(self, input_size):
        super(MappingNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)

        # He 初始化
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def scale_output(self, x, min_value, max_value):
        return min_value + (max_value - min_value) * torch.sigmoid(x)  # 缩放输出到[min_value, max_value]

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        x = self.scale_output(x, 2, 7)
        return x

# 创建模型实例
model = MappingNN(256 * 256 * 3)  # 输入大小应为256 * 256 * 3
model.load_state_dict(torch.load('mapping_nn_model1.pth'))
model.eval()

def preprocess_image(image_path):
    input_img = Image.open(image_path).convert('RGB')
    input_img = input_img.resize((256, 256))  # 调整图像大小
    img_np = np.asarray(input_img)
    return img_np.astype(np.float32) / 255.0  # 返回归一化后的图像

# 创建图像对比的文件夹
if not os.path.exists('comparison_images'):
    os.makedirs('comparison_images')

# 获取所有图像文件
image_files = glob.glob('input_images/*')  # 获取 input_images 下的所有文件

results = []

for image_path in image_files:
    processed_image = preprocess_image(image_path)

    # 计算 MOS 值
    mos_original = predict_mos(processed_image)

    # 将图像输入模型进行预测
    inputs = torch.tensor(processed_image.flatten(), dtype=torch.float32).unsqueeze(0)
    output_value = model(inputs)

    # 生成增强图像
    img = Image.open(image_path).convert('RGB')  # 读取图像并转换为RGB
    img_np = np.array(img)  # 转换为numpy数组
    new_image = illumination_boost(img_np, 7 - output_value.item())

    # 计算增强后图像的 MOS 值
    mos_enhanced = predict_mos(new_image)

    # 记录结果
    results.append({
        'Image Name': os.path.basename(image_path),
        'Predicted Output Value': output_value.item(),
        'Original Image MOS': mos_original,
        'Enhanced Image MOS': mos_enhanced
    })

    # 拼接图像
    img = Image.open(image_path).convert('RGB')
    enhanced_image = Image.fromarray(new_image)

    # 创建新的空图像，保持原始尺寸
    combined_width = img.width + enhanced_image.width
    combined_height = max(img.height, enhanced_image.height)
    combined_image = Image.new('RGB', (combined_width, combined_height))

    # 将原始图像和增强后的图像粘贴到新图像上
    combined_image.paste(img, (0, 0))
    combined_image.paste(enhanced_image, (img.width, 0))

    # 保存拼接后的图像
    original_filename = os.path.splitext(os.path.basename(image_path))[0]
    combined_image.save(f'comparison_images/combined_image_{original_filename}.jpg')

# 创建 DataFrame 并保存为 CSV
results_df = pd.DataFrame(results)
results_df.to_csv('enhancement_results.csv', index=False)
