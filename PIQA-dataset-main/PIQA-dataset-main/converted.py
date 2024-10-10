import glob
import numpy as np
import os
from tensorflow import keras
from PIL import Image
import pandas as pd

np.random.seed(0)

test_dir = './PIQA_dataset/images/'
output_dir = './results/'

# 加载转换后的 .keras 模型
loaded = keras.models.load_model("checkpoints/PIQA.keras")

# 读取 MOS 数据
mos_df = pd.read_csv('./PIQA_dataset/X_test.csv')
mos_np = np.array(mos_df)
nam_dict = {i[0]: i[1] for i in mos_np}  # 简化字典创建

print(len(nam_dict))

prediction_list = []

for key in nam_dict:
    print(key)
    input_img = Image.open(test_dir + key)
    img_np = np.asarray(input_img)
    H, W = img_np.shape[0], img_np.shape[1]

    # 随机裁剪 256x256 的区域
    xx = np.random.randint(0, W - 256)
    yy = np.random.randint(0, H - 256)
    img = img_np[yy:yy + 256, xx:xx + 256, :]

    test_img = np.expand_dims(np.float32(img / 255.0), axis=0)
    pred_test = loaded.predict(test_img)
    print('file_name =', key, 'MOS =', pred_test)

    prediction_list.append([key, np.squeeze(pred_test)])

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 保存结果到 CSV 文件
data = pd.DataFrame(prediction_list)
data.to_csv(output_dir + 'results.csv', header=True, index=False)
