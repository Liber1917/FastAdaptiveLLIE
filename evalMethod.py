import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from evalPIQAtest import predict_mos
import glob
from HE_Based.illumination_boost import illumination_boost

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

    def scale_output(self,x, min_value, max_value):
        return min_value + (max_value - min_value) * torch.sigmoid(x)  # 缩放输出到[min_value, max_value]

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        x = self.scale_output(x,2,7)
        return x


# 创建模型实例
model = MappingNN(256 * 256 * 3)  # 输入大小应为256 * 256 * 3
# 假设你已经定义了 MappingNN 类和 illumination_boost 函数，并加载了模型
model.load_state_dict(torch.load('mapping_nn_model1.pth'))
model.eval()


# 读取图像名称
eval_data = pd.read_csv('PIQA-dataset-main/PIQA-dataset-main/PIQA_dataset/X_eval.csv')
image_names = eval_data.iloc[:, 0].tolist()  # 获取第一列的图像名称

results = []

def preprocess_image(image_path):
    input_img = Image.open(image_path).convert('RGB')
    input_img = input_img.resize((256, 256))  # 调整图像大小
    img_np = np.asarray(input_img)
    return img_np.astype(np.float32) / 255.0  # 返回归一化后的图像

for image_name in image_names:
    image_path = f'PIQA-dataset-main/PIQA-dataset-main/PIQA_dataset/images/{image_name}'
    processed_image = preprocess_image(image_path)

    # 计算 MOS 值
    mos_original = predict_mos(processed_image)

    # 将图像输入模型进行预测
    inputs = torch.tensor(processed_image.flatten(), dtype=torch.float32).unsqueeze(0)
    output_value = model(inputs)

    # 生成增强图像
    new_image = illumination_boost(processed_image, 7-output_value.item())

    # 计算增强后图像的 MOS 值
    mos_enhanced = predict_mos(new_image)

    # 记录结果
    results.append({
        'Image Name': image_name,
        'Predicted Output Value': output_value.item(),
        'Original Image MOS': mos_original,
        'Enhanced Image MOS': mos_enhanced
    })

# 创建 DataFrame 并保存为 CSV
results_df = pd.DataFrame(results)
results_df.to_csv('enhancement_results.csv', index=False)
