import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from evalPIQAtest import predict_mos  # 导入计算 MOS 的函数

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

def preprocess_image(image_path):
    input_img = Image.open(image_path)
    img_np = np.asarray(input_img)
    H, W, _ = img_np.shape
    xx = np.random.randint(0, W - 256)
    yy = np.random.randint(0, H - 256)
    img = img_np[yy:yy + 256, xx:xx + 256, :]
    img = np.float32(img / 255.0)  # 归一化
    return img.flatten()  # 返回展平后的图像特征向量

import glob


# 指定文件夹路径
image_folder = 'PIQA-dataset-main/PIQA-dataset-main/PIQA_dataset/images/'
# 使用 glob 获取所有图片文件
image_list = glob.glob(image_folder + '*.png')  # 假设图片格式为 PNG

# criterion = nn.MSELoss()  # 定义均方误差损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器，设置学习率为0.001

from HE_Based.illumination_boost import illumination_boost
total_images = len(image_list)  # 总图像数量
for idx,image_path in enumerate(image_list):  # 遍历所有图像
    processed_image = preprocess_image(image_path)  # 预处理图像
    mos0 = predict_mos(np.array(processed_image).reshape(256, 256, 3))  # 计算原始图像的MOS值

    inputs = torch.tensor(processed_image, dtype=torch.float32).unsqueeze(0)  # 将特征转换为张量，并增加一个维度（batch size）
    output_value = model(inputs)  # 将输入特征传入模型，得到输出值

    new_image = illumination_boost(processed_image.reshape(256, 256, 3),(output_value.item()))  # 根据输出值生成新图像
    mos1 = predict_mos(np.array(new_image))  # 计算生成图像的MOS值

    # target_value = mos1 - mos0  # 计算目标值，作为损失的依据
    target_value = torch.tensor(mos1 - mos0, dtype=torch.float32, requires_grad=True)


    # 训练
    optimizer.zero_grad()  # 清空之前的梯度信息
    # loss = criterion(output_value, torch.tensor([[target_value]], dtype=torch.float32))  # 计算损失
    loss = torch.abs(target_value) + 1e-6 #
    loss.backward()  # 反向传播，计算梯度
    optimizer.step()  # 更新模型参数

    print(f'Processing image {idx + 1}/{total_images}')  # 输出当前处理的图片序号和总数

# 保存模型
torch.save(model.state_dict(), 'mapping_nn_model1.pth')  # 保存模型权重
