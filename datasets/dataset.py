import os
import torch
import torch.utils.data
from PIL import Image
from datasets.data_augment import PairCompose, PairToTensor, PairRandomHorizontalFilp
import numpy as np
import imageio
import random

def remove_black_level(img, black_level=63, white_level=4*255):
    img = np.maximum(img-black_level, 0) / (white_level-black_level)
    return img

class LLdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self):
        train_dataset = AllWeatherDataset(self.config.data.data_dir,
                                          patch_size=self.config.data.patch_size,
                                          filelist='train.txt')
        val_dataset = AllWeatherDataset(self.config.data.data_dir,
                                        patch_size=self.config.data.patch_size,
                                        filelist='val.txt', train=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class AllWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, filelist=None, train=True):
        super().__init__()

        self.dir = dir
        self.file_list = filelist

        self.train = train

        self.train_list = os.path.join(dir, self.file_list)
        with open(self.train_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]

        self.input_names = input_names
        self.patch_size = patch_size
        if 'ZRR' in self.dir:
            self.black_level, self.white_level, self.rho = 63, 1020, 8
        elif 'MAI' in self.dir:
            self.black_level, self.white_level, self.rho = 255, 4095, 8
        elif 'MI' in self.dir:
            self.black_level, self.white_level = 0, None
        elif 'Z6' in self.dir:
            self.black_level, self.white_level = 0, None
        else:
            raise ValueError('Get wrong dataset name:{}'.format(dir))    
        self.valid_indices = []

        for i, line in enumerate(self.input_names):
            inp, gt = line.split(' ')[:2]
            h_in, w_in = imageio.imread(inp).shape[:2]
            h_gt, w_gt = imageio.imread(gt).shape[:2]

            if min(h_in, h_gt) >= self.patch_size and min(w_in, w_gt) >= self.patch_size:
                self.valid_indices.append(i)

        print(f"[Z6] valid samples: {len(self.valid_indices)} / {len(self.input_names)}")



    def random_crop(self, input_img, gt_img, gt_img_gray):
        _, h_in, w_in = input_img.shape
        _, h_gt, w_gt = gt_img.shape

        h = min(h_in, h_gt)
        w = min(w_in, w_gt)

        if h < self.patch_size or w < self.patch_size:
            return None

        x0 = random.randint(0, h - self.patch_size)
        y0 = random.randint(0, w - self.patch_size)

        input_crop = input_img[:, x0:x0+self.patch_size, y0:y0+self.patch_size]
        gt_crop = gt_img[:, x0:x0+self.patch_size, y0:y0+self.patch_size]
        gt_gray_crop = gt_img_gray[:, x0:x0+self.patch_size, y0:y0+self.patch_size]

        return (
            input_crop.contiguous(),
            gt_crop.contiguous(),
            gt_gray_crop.contiguous(),
        )


    def get_images(self, index):
        name = self.input_names[index].replace('\n', '')
        input_name = name.split(' ')[0]
        img_id = input_name.split('/')[-1].split('.')[0]

        if 'ZRR' in self.dir and self.train:
            gt_name = name.split(' ')[2]
        else:
            gt_name = name.split(' ')[1]

        input_raw = np.expand_dims(imageio.imread(input_name), axis=-1)
        gt_img = imageio.imread(gt_name)
        gt_img_gray = np.expand_dims(imageio.imread(gt_name, mode='L'), axis=-1)

        if ('MI' in self.dir) or ('Z6' in self.dir):
            input_raw = input_raw.astype(np.float32)
            input_raw /= (np.max(input_raw) + 1e-6)
        else:
            input_raw = np.maximum(input_raw - self.black_level, 0) / \
                        (self.white_level - self.black_level)


        input_raw = torch.tensor(input_raw).permute(2, 0, 1).float()
        gt_img = torch.tensor(gt_img / 255.0).permute(2, 0, 1).float()
        gt_img_gray = torch.tensor(gt_img_gray / 255.0).permute(2, 0, 1).float()

        # train / val 공통 patch crop
        out = self.random_crop(input_raw, gt_img, gt_img_gray)
        if out is None:
            raise RuntimeError(f"Image too small for patch_size: {img_id}")

        input_raw, gt_img, gt_img_gray = out

        return input_raw, gt_img, gt_img_gray, img_id



    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        return self.get_images(real_idx)
