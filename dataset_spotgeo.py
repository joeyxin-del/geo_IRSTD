from utils import *
import matplotlib.pyplot as plt
import os
import time
import albumentations
import logging
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

logger = logging.getLogger(__name__)


IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')

class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None, use_filtered_data=False):
        super(TrainSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + dataset_name
        self.patch_size = patch_size
        
        # Choose between original and filtered index files
        if use_filtered_data:
            index_file = self.dataset_dir+'/img_idx/train_' + dataset_name + '_filtered.txt'
            logger.info(f"Using filtered training data: {index_file}")
        else:
            index_file = self.dataset_dir+'/img_idx/train_' + dataset_name + '.txt'
            logger.info(f"Using original training data: {index_file}")
            
        with open(index_file, 'r') as f:
            self.train_list = f.read().splitlines()
        # if img_norm_cfg == None:
        #     self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        # else:
        #     self.img_norm_cfg = img_norm_cfg

        self.tranform = augumentation()
        self.Resize = albumentations.Resize(patch_size, patch_size)
        
        # Pre-cache file extension and build file paths
        self._cache_file_info()
        
    def _cache_file_info(self):
        """Cache file extension and pre-build file paths for faster loading"""
        img_list = os.listdir(self.dataset_dir + '/images/')
        self.img_ext = os.path.splitext(img_list[0])[-1]
        if not self.img_ext in IMG_EXTENSIONS:
            raise TypeError("Unrecognized image extensions.")
        
        # Pre-build file paths
        self.img_paths = []
        self.mask_paths = []
        for filename in self.train_list:
            img_path = os.path.join(self.dataset_dir, 'images', filename + self.img_ext)
            mask_path = os.path.join(self.dataset_dir, 'masks', filename + self.img_ext)
            self.img_paths.append(img_path)
            self.mask_paths.append(mask_path)
        
    def __getitem__(self, idx):
        # print(f"Processing image {idx} of {len(self.train_list)}")
        data_load_start = time.time()
        
        # Use pre-built paths instead of constructing them each time
        img = Image.open(self.img_paths[idx]).convert('I')
        mask = Image.open(self.mask_paths[idx])
        
        # img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        img = Normalized(np.array(img, dtype=np.float32))
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape)>2:
            mask = mask[:,:,0]

        # Resized = self.Resize(image=img, mask=mask)
        # img = Resized["image"]
        # mask = Resized["mask"]
        
        # img_patch, mask_patch = random_crop(img, mask, self.patch_size)
        img_patch, mask_patch = random_crop(img, mask, self.patch_size)
        img_patch, mask_patch = self.tranform(img_patch, mask_patch)
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]  # (255,255)-->(1,255,255)
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
        
        # Remove debug print to reduce overhead
        # data_load_end = time.time()
        # data_load_time = data_load_end - data_load_start
        # print(f"Data load time1111111111111111111: {data_load_time:.2f}s")
        
        return img_patch, mask_patch
    def __len__(self):
        return len(self.train_list)

class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, patch_size, img_norm_cfg=None, use_filtered_data=False):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name      #datasets/IRSTD-1K
        
        # Choose between original and filtered index files
        if use_filtered_data:
            index_file = self.dataset_dir+'/img_idx/test_' + test_dataset_name + '_filtered.txt'
            logger.info(f"Using filtered test data: {index_file}")
        else:
            index_file = self.dataset_dir+'/img_idx/test_' + test_dataset_name + '.txt'
            logger.info(f"Using original test data: {index_file}")
            
        with open(index_file, 'r') as f:
            self.test_list = f.read().splitlines()
        self.patch_size = patch_size
        self.Resize = albumentations.Resize(patch_size, patch_size)
        # if img_norm_cfg == None:
        #     self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        # else:
        #     self.img_norm_cfg = img_norm_cfg
            # test data augmentations
            # self.aug = albumentations.Compose([
            #     albumentations.RandomResizedCrop(256, 256),
            #     albumentations.Transpose(p=0.5),
            #     albumentations.HorizontalFlip(p=0.5),
            #     albumentations.VerticalFlip(p=0.5),
            #     albumentations.HueSaturationValue(
            #         hue_shift_limit=0.2,
            #         sat_shift_limit=0.2,
            #         val_shift_limit=0.2,
            #         p=0.5
            #     ),
            #     albumentations.RandomBrightnessContrast(
            #         brightness_limit=(-0.1, 0.1),
            #         contrast_limit=(-0.1, 0.1),
            #         p=0.5
            #     ),
            #     albumentations.Normalize(
            #         mean=[0.485, 0.456, 0.406],
            #         std=[0.229, 0.224, 0.225],
            #         max_pixel_value=255.0,
            #         p=1.0
            #     )
            # ], p=1.)
        
        # Pre-cache file extension and build file paths
        self._cache_file_info()
        
    def _cache_file_info(self):
        """Cache file extension and pre-build file paths for faster loading"""
        img_list = os.listdir(self.dataset_dir + '/images/')
        self.img_ext = os.path.splitext(img_list[0])[-1]
        if not self.img_ext in IMG_EXTENSIONS:
            raise TypeError("Unrecognized image extensions.")
        
        # Pre-build file paths
        self.img_paths = []
        self.mask_paths = []
        for filename in self.test_list:
            img_path = os.path.join(self.dataset_dir, 'images', filename + self.img_ext)
            mask_path = os.path.join(self.dataset_dir, 'masks', filename + self.img_ext)
            self.img_paths.append(img_path)
            self.mask_paths.append(mask_path)
        
    def __getitem__(self, idx):
        # print(self.img_paths[idx])
        # test_load_start = time.time()
        # Use pre-built paths instead of constructing them each time
        img = Image.open(self.img_paths[idx]).convert('I')
        mask = Image.open(self.mask_paths[idx])
        
        # img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        img = Normalized(np.array(img, dtype=np.float32))
        mask = np.array(mask, dtype=np.float32)  / 255.0
        if len(mask.shape)>2:
            mask = mask[:,:,0]

        # Resized = self.Resize(image=img, mask=mask)
        # img = Resized["image"]
        # mask = Resized["mask"]
        # img, mask = pad(img, mask, self.patch_size)
        h, w = img.shape
        img = PadImg(img, 32)
        mask = PadImg(mask, 32)
        
        img, mask = img[np.newaxis,:], mask[np.newaxis,:]
        
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        # test_load_end = time.time()
        # test_load_time = test_load_end - test_load_start
        # print(f"Test load time: {test_load_time:.2f}s")
        
        return img, mask, [h, w], self.test_list[idx]
    def __len__(self):
        return len(self.test_list)

class EvalSetLoader(Dataset):
    def __init__(self, dataset_dir, mask_pred_dir, test_dataset_name, model_name):
        super(EvalSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.mask_pred_dir = mask_pred_dir
        self.test_dataset_name = test_dataset_name
        self.model_name = model_name
        with open(self.dataset_dir+'/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()

    def __getitem__(self, idx):
        img_list_pred = os.listdir(self.mask_pred_dir + self.test_dataset_name + '/' + self.model_name + '/')
        img_ext_pred = os.path.splitext(img_list_pred[0])[-1]

        img_list_gt = os.listdir(self.dataset_dir + '/masks/')
        img_ext_gt = os.path.splitext(img_list_gt[0])[-1]
        
        if not img_ext_gt in IMG_EXTENSIONS:
            raise TypeError("Unrecognized GT image extensions.")
        if not img_ext_pred in IMG_EXTENSIONS:
            raise TypeError("Unrecognized Predicted image extensions.")
        mask_pred = Image.open((self.mask_pred_dir + self.test_dataset_name + '/' + self.model_name + '/' + self.test_list[idx] + img_ext_pred).replace('//','/'))
        mask_gt = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + img_ext_gt).replace('//','/'))
                
        mask_pred = np.array(mask_pred, dtype=np.float32)  / 255.0
        mask_gt = np.array(mask_gt, dtype=np.float32)  / 255.0
        if len(mask_pred.shape)>3:
            mask_pred = mask_pred[:,:,0]
        if len(mask_gt.shape)>3:
            mask_gt = mask_gt[:,:,0]
            
        # Get height and width from the mask array
        shape = mask_pred.shape
        h, w = shape[0], shape[1]
        
        mask_pred, mask_gt = mask_pred[np.newaxis,:], mask_gt[np.newaxis,:]
        
        mask_pred = torch.from_numpy(np.ascontiguousarray(mask_pred))
        mask_gt = torch.from_numpy(np.ascontiguousarray(mask_gt))
        return mask_pred, mask_gt, [h,w]
    def __len__(self):
        return len(self.test_list) 


class augumentation(object):
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[::-1, :]
            target = target[::-1, :]
        if random.random()<0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
        if random.random()<0.5:
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
        return input, target
