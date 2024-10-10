import numpy as np
from PIL import Image
from scipy.special import erf

def illumination_boost(X, lambda_):
    """
    Enhances an image illumination using the method proposed at
    Nighttime image enhancement using a new illumination boost algorithm
    by Al-Ameen, Zohair.

    Parameters:
      X: RGB Image (numpy array)
      lambda_: Factor used in the equations I3 and I4
    """
    # Convert image to float32, even if it is already a float
    X = X.astype(np.float32) / 255.0

    # Process the image with a logarithmic scaling function
    I1 = (np.max(X) / np.log(np.max(X) + 1)) * np.log(X + 1)

    # Use a non-complex exponential function to modify the local contrast
    I2 = 1 - np.exp(-X)

    # Use a LIP model, adapting it to the nature of the image
    I3 = (I1 + I2) / (lambda_ + (I1 * I2))

    # Compute a modified CDF-HSD function to increase brightness of dark regions
    I4 = erf(lambda_ * np.arctan(np.exp(I3)) - 0.5 * I3)

    # Apply normalization function
    I5 = (I4 - np.min(I4)) / (np.max(I4) - np.min(I4))

    return (I5 * 255).astype(np.uint8)

# Example usage:
# img_path = 'input_images/airplane.jpg'
# img = Image.open(img_path).convert('RGB')  # 读取图像并转换为RGB
# img_np = np.array(img)  # 转换为numpy数组

# enhanced_img = illumination_boost(img_np, 3)  # 增强图像
# enhanced_image = Image.fromarray(enhanced_img)  # 转换回PIL图像
# enhanced_image.save('enhanced_image.jpg')  # 保存增强后的图像
