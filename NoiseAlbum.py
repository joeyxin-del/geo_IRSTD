import numpy as np
import cv2

# 添加高斯噪声
def add_gaussian_noise(image, mean=0, var=1, strength=1, **kwargs):
    # 计算标准差
    sigma = var ** 0.5
    # 生成与图像形状相同的高斯噪声
    gauss = np.random.normal(mean, sigma, image.shape).astype('uint8')/255

    noisy = image + gauss * strength

    # 确保噪声加成后图像的值在合理范围内
    noisy = np.clip(noisy, 0, 1)  # 保证像素值在[0, 1]范围内
    return noisy.astype(np.float32)

# 添加均匀噪声
def add_uniform_noise(image, low=0, high=255, strength=1, **kwargs):
    uniform = np.random.uniform(low, high, image.shape).astype('uint8')/255

    noisy =image + uniform * strength

    # 确保噪声加成后图像的值在合理范围内
    noisy = np.clip(noisy, 0, 1)  # 保证像素值在[0, 1]范围内

    return noisy.astype(np.float32)


# 添加椒盐噪声
def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01, strength=1, **kwargs):
    # 创建噪声图像的副本
    noisy = image.copy()
    # 计算总像素数
    total_pixels = image.shape[0] * image.shape[1]
    # 添加盐噪声（白点）
    num_salt = int(total_pixels * salt_prob * strength)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 1
    # 添加椒噪声（黑点）
    num_pepper = int(total_pixels * pepper_prob * strength)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0

    # 确保噪声加成后图像的值在合理范围内
    noisy = np.clip(noisy, 0, 1)  # 保证像素值在[0, 1]范围内

    return noisy.astype(np.float32)

# 添加线性变换噪声
def add_linear_transform_noise(image, strength=1, **kwargs):
    # 复制输入图像
    x_copy = image.copy()

    # 获取图像的形状
    h, w = image.shape[:2]

    # 循环移位：将图像的第一行移到最后
    x_copy = np.roll(x_copy, shift=1, axis=0)  # 按行移位
    # 或者可以按列进行移位
    # x_copy = np.roll(x_copy, shift=1, axis=1)  # 按列移位

    # 生成噪声：计算移位后的图像与原图像的差异
    noise = x_copy - image

    # 调整噪声强度，使用 alpha 和 beta 控制
    noisy = image +   noise * strength

    # 确保噪声加成后图像的值在合理范围内
    noisy = np.clip(noisy, 0, 1)  # 保证像素值在[0, 1]范围内

    return noisy
