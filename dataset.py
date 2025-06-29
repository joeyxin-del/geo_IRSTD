from utils import *  # 导入自定义的工具函数或类
import matplotlib.pyplot as plt  # 导入绘图库
import os  # 导入处理操作系统相关功能的库
import albumentations as A  # 导入数据增强库 Albumentations

from NoiseAlbum import *
import albumentations.augmentations.functional as F  # 导入 Albumentations 的功能模块
from albumentations.pytorch import ToTensorV2  # 导入将 Albumentations 转换为 PyTorch 张量的工具

from scipy import ndimage  # 导入 SciPy 库中的图像处理功能
import torchvision.utils as vutils  # 导入 PyTorch 视觉工具

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 设置环境变量，允许 OpenMP 重复加载库

# 支持的图像文件扩展名
IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')


# 训练集数据加载器类
class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size , img_norm_cfg=None):
        super(TrainSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + dataset_name  # 数据集目录
        self.patch_size = patch_size  # 图像裁剪尺寸
        self.noisestrength = 1
        with open(self.dataset_dir + '/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()  # 读取训练集列表
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)  # 获取图像归一化配置
        else:
            self.img_norm_cfg = img_norm_cfg  # 使用传入的图像归一化配置
        # 图像增强的处理流程定义
        self.transform = A.Compose(
            [
                # A.Resize(patch_size, patch_size),  # 图像尺寸调整
                A.OneOf([A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),  # 随机平移、缩放、旋转
                         A.RandomRotate90(p=0.5),  # 随机90度旋转
                         A.Flip(p=0.5),  # 随机翻转
                         A.ElasticTransform(p=0.1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03), ], p=0.7),

                A.OneOf([A.Sharpen(p=.5),  # 锐化
                         A.RandomBrightnessContrast(p=0.2), ]),  # 随机亮度和对比度
                A.OneOf([
                    A.MotionBlur(p=.2),  # 运动模糊
                    A.MedianBlur(blur_limit=3, p=0.1),  # 中值模糊
                    A.Blur(blur_limit=3, p=0.1), ], p=0.3),  # 普通模糊

            ]
        )

        # self.transform = A.Compose([
        #     A.OneOf([
        #         A.Lambda(image=lambda image, **kwargs: add_gaussian_noise(image, strength=self.noisestrength, **kwargs)),
        #         A.Lambda(image=lambda image, **kwargs: add_uniform_noise(image, strength=self.noisestrength, **kwargs)),
        #         A.Lambda(image=lambda image, **kwargs: add_salt_and_pepper_noise(image, strength=self.noisestrength, **kwargs)),
        #         A.Lambda(image=lambda image, **kwargs: add_linear_transform_noise(image, strength=self.noisestrength, **kwargs)),
        #     ], p=1.0)
        #     ])

    def __getitem__(self, idx):
        img_list = os.listdir(self.dataset_dir + '/images/')  # 获取图像文件列表
        img_ext = os.path.splitext(img_list[0])[-1]  # 获取图像文件扩展名
        if not img_ext in IMG_EXTENSIONS:
            raise TypeError("Unrecognized image extensions.")  # 如果图像扩展名不在支持列表中，则报错
        img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + img_ext).replace('//', '/')).convert(
            'I')  # 打开图像并转换为灰度图像
        mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + img_ext).replace('//', '/')).convert(
            'I')  # 打开对应的标签图像
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)  # 对图像进行归一化处理
        mask = np.array(mask, dtype=np.float32) / 255.0  # 将标签图像转换为numpy数组并归一化到[0,1]范围
        if len(mask.shape) > 3:
            mask = mask[:, :, 0]  # 如果标签图像的通道数大于3，则取第一个通道

        # 图像增强处理
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        # 将图像和标签转换为PyTorch张量
        img_patch, mask_patch = img[np.newaxis, :], mask[np.newaxis, :]
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))

        return img_patch, mask_patch

    def __len__(self):
        return len(self.train_list)  # 返回训练集样本数量


# 测试集数据加载器类
class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, patch_size = 512, img_norm_cfg=None):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name  # 数据集目录
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()  # 读取测试集列表
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)  # 获取训练集的图像归一化配置
        else:
            self.img_norm_cfg = img_norm_cfg  # 使用传入的图像归一化配置

        # 图像尺寸调整的处理流程
        self.transform = A.Compose(
            [
                A.Resize(patch_size, patch_size)  # 调整图像尺寸

            ]
        )

    def __getitem__(self, idx):
        img_list = os.listdir(self.dataset_dir + '/images/')  # 获取图像文件列表
        img_ext = os.path.splitext(img_list[0])[-1]  # 获取图像文件扩展名
        if not img_ext in IMG_EXTENSIONS:
            raise TypeError("Unrecognized image extensions.")  # 如果图像扩展名不在支持列表中，则报错
        img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + img_ext).replace('//', '/')).convert(
            'I')  # 打开图像并转换为灰度图像
        mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + img_ext).replace('//', '/')).convert(
            'I')  # 打开对应的标签图像
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)  # 对图像进行归一化处理
        mask = np.array(mask, dtype=np.float32) / 255.0  # 将标签图像转换为numpy数组并归一化到[0,1]范围
        if len(mask.shape) > 3:
            mask = mask[:, :, 0]  # 如果标签图像的通道数大于3，则取第一个通道

        h, w = img.shape
        img = PadImg(img, 32)  # 对图像进行填充
        mask = PadImg(mask, 32)  # 对标签进行填充

        # 图像尺寸调整处理
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        img, mask = img[np.newaxis, :], mask[np.newaxis, :]
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, mask, [h, w], self.test_list[idx]  # 返回处理后的图像、标签、原始图像尺寸和图像文件名

    def __len__(self):
        return len(self.test_list)  # 返回测试集样本数量


# 评估集数据加载器类
class EvalSetLoader(Dataset):
    def __init__(self, dataset_dir, mask_pred_dir, test_dataset_name, model_name):
        super(EvalSetLoader).__init__()
        self.dataset_dir = dataset_dir  # 数据集目录
        self.mask_pred_dir = mask_pred_dir  # 预测标签的目录
        self.test_dataset_name = test_dataset_name  # 测试集名称
        self.model_name = model_name  # 模型名称
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()  # 读取测试集列表

    def __getitem__(self, idx):
        img_list_pred = os.listdir(
            self.mask_pred_dir + self.test_dataset_name + '/' + self.model_name + '/')  # 获取预测标签的文件列表
        img_ext_pred = os.path.splitext(img_list[0])[-1]  # 获取预测标签的文件扩展名

        img_list_gt = os.listdir(self.dataset_dir + '/masks/')  # 获取真实标签的文件列表
        img_ext_gt = os.path.splitext(img_list[0])[-1]  # 获取真实标签的文件扩展名

        if not img_ext_gt in IMG_EXTENSIONS:
            raise TypeError("Unrecognized GT image extensions.")  # 如果真实标签的扩展名不在支持列表中，则报错
        if not img_ext_pred in IMG_EXTENSIONS:
            raise TypeError("Unrecognized Predicted image extensions.")  # 如果预测标签的扩展名不在支持列表中，则报错
        mask_pred = Image.open((self.mask_pred_dir + self.test_dataset_name + '/' + self.model_name + '/' +
                                self.test_list[idx] + img_ext_pred).replace('//', '/'))  # 打开预测标签
        mask_gt = Image.open(
            (self.dataset_dir + '/masks/' + self.test_list[idx] + img_ext_gt).replace('//', '/'))  # 打开真实标签

        mask_pred = np.array(mask_pred, dtype=np.float32) / 255.0  # 将预测标签转换为numpy数组并归一化到[0,1]范围
        mask_gt = np.array(mask_gt, dtype=np.float32) / 255.0  # 将真实标签转换为numpy数组并归一化到[0,1]范围
        if len(mask_pred.shape) > 3:
            mask_pred = mask_pred[:, :, 0]  # 如果预测标签的通道数大于3，则取第一个通道
        if len(mask_gt.shape) > 3:
            mask_gt = mask_gt[:, :, 0]  # 如果真实标签的通道数大于3，则取第一个通道

        h, w = mask_pred.shape

        mask_pred, mask_gt = mask_pred[np.newaxis, :], mask_gt[np.newaxis, :]  # 将预测标签和真实标签转换为PyTorch张量

        mask_pred = torch.from_numpy(np.ascontiguousarray(mask_pred))
        mask_gt = torch.from_numpy(np.ascontiguousarray(mask_gt))
        return mask_pred, mask_gt, [h, w]  # 返回处理后的预测标签、真实标签和尺寸信息

    def __len__(self):
        return len(self.test_list)  # 返回测试集样本数量


# 自定义的图像增强类
class augumentation(object):
    def __call__(self, input, target):
        if random.random() < 0.5:
            input = input[::-1, :]  # 图像垂直翻转
            target = target[::-1, :]  # 标签垂直翻转
        if random.random() < 0.5:
            input = input[:, ::-1]  # 图像水平翻转
            target = target[:, ::-1]  # 标签水平翻转
        if random.random() < 0.5:
            input = input.transpose(1, 0)  # 图像转置
            target = target.transpose(1, 0)  # 标签转置
        return input, target
