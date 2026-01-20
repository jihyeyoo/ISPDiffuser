import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import utils
from utils.ISP import ISP_process
from models.ddm import GaussianDiffusion, Unet
from models.basic_model import AE, HCCM
from utils.get_canny import Net as CannyFilter
from utils.logging import Adder
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips

def lpips_norm(img, device='cuda'):
	img = img[:, :, :, np.newaxis].transpose((3, 2, 0, 1))
	img = img / (255. / 2.) - 1
	return torch.Tensor(img).to(device)


def calc_lpips(out, target, loss_fn_alex):
	lpips_out = lpips_norm(out)
	lpips_target = lpips_norm(target)
	LPIPS = loss_fn_alex(lpips_out, lpips_target)
	return LPIPS.detach().cpu().item()

class TextureLoss(nn.Module):
    def __init__(self, device='cuda'):
        super(TextureLoss, self).__init__()
        self.filter = CannyFilter(use_cuda=True).to(device)
        self.l1_loss = torch.nn.L1Loss()
    def forward(self, pred_img, target):
        # print('pred_img:', pred_img.shape)
        # print('target', target.shape)
        pred_canny = self.filter(pred_img)
        target_canny = self.filter(target)

        return self.l1_loss(pred_canny, target_canny)
            

class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


class Net(nn.Module):
    def __init__(self, args, config):
        super(Net, self).__init__()

        self.args = args
        self.config = config
        self.device = config.device


        unet = Unet(dim=64, channels=3)
        self.diffusion_model = GaussianDiffusion(unet, image_size=64, objective='pred_x0', sampling_timesteps=25)

        # if self.args.mode == 'training':
        #     self.AE_Downsampler = self.load_stage1(AE_Downsampler(), 'ckpt/AE_Downsampler')
        # else:
        self.AE= AE(channels=64)

    @staticmethod
    def load_stage1(model, model_dir):
        checkpoint = utils.logging.load_checkpoint(os.path.join(model_dir, 'stage1_weight.pth.tar'), 'cuda')
        model.load_state_dict(checkpoint['model'], strict=True)
        return model

    def forward(self, x, y_gray=None, y=None):
        data_dict = {}
        output = self.AE(x,y_gray=y_gray, y=None, pred_y_gray=None)
        gt_gray_fea_down, raw_fea_down = output['gt_gray_fea_down'], output['raw_fea_down']
        data_dict['gt_gray_fea'] = gt_gray_fea_down #diff loss

        
        raw_fea_down = utils.data_transform(raw_fea_down)
        b,c,_,_ = raw_fea_down.shape

        if self.training:
            assert gt_gray_fea_down is not None
            gt_gray_fea_down_norm = utils.data_transform(gt_gray_fea_down)
            pred_gt_gray_fea_down = self.diffusion_model(gt_gray_fea_down_norm, x_cond=raw_fea_down)
        else:
            pred_gt_gray_fea_down = self.diffusion_model.sample(batch_size=b,x_cond=raw_fea_down)
        pred_gt_gray_fea_down = utils.inverse_data_transform(pred_gt_gray_fea_down)

        output = self.AE(x=x,y_gray=None, y=y, pred_y_gray=pred_gt_gray_fea_down)

        recon_gt_img = output['recon_gt_img']
        recon_fea4 = output['recon_gt_fea_ori']
        gt_fea_ori = output['gt_fea_ori']
        pred_hist = output['pred_hist']
        gt_hist = output['gt_hist']


        data_dict["pred_gt_gray_fea"] = pred_gt_gray_fea_down #diff loss
        data_dict['recon_gt_fea_ori'] = recon_fea4 #feature loss
        data_dict['gt_fea_ori'] = gt_fea_ori
        data_dict['pred_hist'] = pred_hist #hist loss
        data_dict['gt_hist']=gt_hist
        data_dict["recon_gt_img"] = recon_gt_img #recon loss

        return data_dict


class ISPDiffuser(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = Net(args, config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.texture_loss = TextureLoss()

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0
        if os.path.isfile(args.resume):
            self.load_ddm_ckpt(args.resume, ema=False)
            self.model.eval()
        else:
            print('Pre-trained model path is missing!')

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        state_dict = checkpoint['state_dict']

        self.model.load_state_dict(state_dict, strict=True)
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint {} step {}".format(load_path, self.step))

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        # for name, param in self.model.named_parameters():
        #     if "AE_Downsampler" in name:
        #         param.requires_grad = False
        #     else:
        #         param.requires_grad = True

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            for i, (x,y,y_gray, img_id) in enumerate(train_loader):
                # import pdb
                # pdb.set_trace()
                # x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                self.model.train()
                self.step += 1

                x, y, y_gray = x.to(self.device), y.to(self.device), y_gray.to(self.device)

                output = self.model(x, y, y_gray)

                diff_loss, feature_loss, content_loss, hist_loss, texture_loss = self.noise_estimation_loss(output,y)
                loss = diff_loss + feature_loss + 0.01* hist_loss + 0.01* texture_loss+content_loss

                data_time += time.time() - data_start

                # revised
                if self.step % 10 == 0:
                    print("step:{}, diff_loss:{:.5f} feature_loss:{:.5f} hist_loss:{:.8f} texture_loss:{:.5f} content_loss:{:.5f}".
                        format(self.step, diff_loss.item(), feature_loss.item(), hist_loss.item(), texture_loss.item(), content_loss.item()))
                                    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0 and self.step != 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step)

                    utils.logging.save_checkpoint({'step': self.step,
                                                   'epoch': epoch + 1,
                                                   'state_dict': self.model.state_dict(),
                                                   'optimizer': self.optimizer.state_dict(),
                                                   'ema_helper': self.ema_helper.state_dict(),
                                                   'params': self.args,
                                                   'config': self.config},
                                                  filename = os.path.join(self.config.data.ckpt_dir, 'model_latest'))

    def noise_estimation_loss(self, output, y):
        gt_gray_fea, recon_gt_gray_fea = output['gt_gray_fea'], output['pred_gt_gray_fea']
        recon_fea, gt_fea = output["recon_gt_fea_ori"], output["gt_fea_ori"]
        pred_hist, gt_hist = output['pred_hist'],output['gt_hist']
        recon_gt_img = output['recon_gt_img']

        diff_loss =  self.l2_loss(recon_gt_gray_fea, gt_gray_fea)
        content_loss = self.l1_loss(recon_gt_img, y)
        feature_loss = self.l1_loss(recon_fea, gt_fea)
        hist_loss = self.l2_loss(pred_hist, gt_hist)
        texture_loss = self.texture_loss(recon_gt_gray_fea, gt_gray_fea)

        
        return diff_loss, feature_loss, content_loss, hist_loss, texture_loss

    def sample_validation_patches(self, val_loader, step):
        self.model.eval()
        loss_fn_alex_v1 = lpips.LPIPS(net='alex', version='0.1').to(self.device)
        psnr_adder = Adder()
        ssim_adder = Adder()
        lpips_adder = Adder()

        with torch.no_grad():
            print('Performing validation at step: {}'.format(step))
            for i, (x, y, y_gray, img_id) in enumerate(val_loader):

                b, _, img_h, img_w = y.shape
                x = x.to(self.device)

                y = y.permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()
                y = np.clip(y * 255.0, 0, 255.0).astype('uint8')

                
                for i in range(10):
                    
                    pred_x = self.model(x)["recon_gt_img"]
                    recon_img = pred_x.permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()
                    recon_img = np.clip(recon_img * 255.0, 0, 255.0).astype('uint8')
                
                    psnr_gt = peak_signal_noise_ratio(recon_img, y, data_range=255)
                    # revised : 11->7, channel_axis
                    ssim_gt = structural_similarity(recon_img, y, win_size=7, data_range=255, channel_axis=-1, gaussian_weights=True)
                    lpips_gt = calc_lpips(recon_img, y, loss_fn_alex_v1)
                    if i == 0:
                        pred_x_save = pred_x
                        psnr_save = psnr_gt
                        ssim_save = ssim_gt
                        lpips_save = lpips_gt
                    elif i>0 and psnr_gt > psnr_save:
                        pred_x_save = pred_x
                        psnr_save = psnr_gt
                        ssim_save = ssim_gt
                        lpips_save = lpips_gt


                utils.logging.save_image(pred_x_save, os.path.join(self.config.sampling.img_save_path, '{}.png'.format(img_id)))

                psnr_adder(psnr_save)
                ssim_adder(ssim_save)
                lpips_adder(lpips_save)
                print('idx:{} psnr:{} ssim: {}, lpips:{}'.format(img_id,  psnr_save, ssim_save, lpips_save))
            print('avg psnr: {}, avg ssim: {} lpips:{}'.format(psnr_adder.average(), ssim_adder.average(), lpips_adder.average()))
