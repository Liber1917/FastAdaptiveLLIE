import numpy as np
from tensorflow import keras
from PIL import Image

# 设置随机种子
np.random.seed(0)

# 加载模型
loaded_model = keras.models.load_model("PIQA-dataset-main/PIQA-dataset-main/checkpoints/PIQA.keras")

def predict_mos(image_array):
    """
    Predict the MOS value from the given image array.
    """
    # 将 numpy 数组转换为 PIL 图像
    input_img = Image.fromarray(image_array.astype(np.uint8))

    # 调整图像大小为 (256, 256)
    input_img = input_img.resize((256, 256))
    # 进行预测
    test_img = np.expand_dims(np.float32(input_img) / 255.0, axis=0)  # 归一化并扩展维度
    pred_test = loaded_model.predict(test_img)
    return np.squeeze(pred_test)  # 返回预测值

# 示例调用
# image_path = './PIQA-dataset-main/PIQA-dataset-main/PIQA_dataset/images/09778.png'
# image_array = np.array(Image.open(image_path).convert("RGB"))  # 读取并转换为 NumPy 数组
# mos_value = predict_mos(image_array)
# print('Predicted MOS:', mos_value)
