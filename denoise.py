import os.path
import logging

import numpy as np
from numpy import linalg as la
from collections import OrderedDict
import scipy.io 

import torch

from utils import utils_logger
from utils import utils_model
from utils import utils_image as util
from Hysime import *
from SAH import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
Spyder (Python 3.7)
PyTorch 1.6.0
Windows 10 or Linux
Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/DPIR
        https://github.com/cszn/IRCNN
        https://github.com/cszn/KAIR
@article{zhang2020plug,
  title={Plug-and-Play Image Restoration with Deep Denoiser Prior},
  author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint},
  year={2020}
}
% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com; homepage: https://cszn.github.io/)
by Kai Zhang (01/August/2020)

# --------------------------------------------
|--model_zoo               # model_zoo
   |--drunet_gray          # model_name, for color images
   |--drunet_color
|--testset                 # testsets
   |--set12                # testset_name
   |--bsd68
   |--cbsd68
|--results                 # results
   |--set12_dn_drunet_gray # result_name = testset_name + '_' + 'dn' + model_name
   |--set12_dn_drunet_color
# --------------------------------------------
"""


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    noise_level_img = 15                 # set AWGN noise level for noisy image
    noise_level_model = noise_level_img  # set noise level for model
    model_name = 'drunet_gray'           # set denoiser model, 'drunet_gray' | 'drunet_color'
    testset_name = 'dataset'               # set test set,  'bsd68' | 'cbsd68' | 'set12'
    x8 = False                           # default: False, x8 to boost performance
    show_img = False                     # default: False
    border = 0                           # shave boader to calculate PSNR and SSIM
    k_subspace = 8                               # k_subspace is the number of selected columns

    if 'color' in model_name:
        n_channels = 3                   # 3 for color image
    else:
        n_channels = 1                   # 1 for grayscale image

    model_pool = 'model_zoo'             # fixed
    testsets = 'testsets'                # fixed
    results = 'results'                  # fixed
    task_current = 'dn'                  # 'dn' for denoising
    result_name = testset_name + '_' + task_current + '_' + model_name

    model_path = os.path.join(model_pool, model_name+'.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(testsets, testset_name) # L_path, for Low-quality images
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from models.network_unet import UNetRes as net
    model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    logger.info('model_name:{}, model sigma:{}, image sigma:{}'.format(model_name, noise_level_img, noise_level_model))
    logger.info(L_path)
    #L_paths = util.get_image_paths(L_path)
    img_path = L_path + "/img_clean_dc.mat"
    mat_data = scipy.io.loadmat(img_path)
    img_clean_dc = mat_data['img_clean_dc']
    hight, width, channel = 150,200,191
    for idx in range(1):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------
        for band in range(channel):
            img = img_clean_dc[:, :, band]
            img_H = img.astype(np.int32)
            # logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
            #img_H = util.imread_uint(img, n_channels=n_channels)
            img_L = util.uint2single(img_H)
            img_L = np.expand_dims(img_L, axis=2)
            img_cat0 = np.concatenate((img_cat0, img_L), axis=2) if band > 0 else img_L
        img_L = img_cat0
        img_name, ext = os.path.splitext(os.path.basename(img_path))
        # Add noise without clipping
        np.random.seed(seed=0)  # for reproducibility
        img_L += np.random.normal(0, noise_level_img/255., img_L.shape)
        Y = img_L.reshape(hight * width, channel)
        w , Rw = est_noise(Y)
        Rw_ori = Rw
        Y = np.sqrt(la.inv(Rw_ori)).dot(Y.T)
        img_L = Y.T.reshape(hight, width, channel)
        Y = img_L.reshape(hight * width, channel)
        w , Rw = est_noise(Y)
        U,sigma,Vt = la.svd(Y.T,full_matrices=False)#svd
        U = U[:, 0:k_subspace]
        eigen_Y_eigencnn = np.empty((k_subspace, hight * width))
        eigen_Y = U.T.dot(Y.T)
        for i in range(k_subspace):
            eigen_im = eigen_Y[i, :]
            min_x = np.min(eigen_im)
            max_x = np.max(eigen_im)
            eigen_im = eigen_im - min_x
            scale = max_x - min_x
            eigen_im = eigen_im.reshape(hight, width) / scale
            sigma_est = np.sqrt(U[:, i].T.dot(Rw).dot(U[:, i])) / scale
            if hight % 2 == 1:
                eigen_im = np.vstack([eigen_im, eigen_im[-1, :]])
            if width % 2 == 1:
                eigen_im = np.hstack([eigen_im, eigen_im[:, -1].reshape(-1,1)])
            eigen_im_single = eigen_im.astype(np.float32)
            if i == 0:
                eigen_im_z1 = eigen_im_single
            input_tensor = np.stack([eigen_im_z1.cpu().numpy() if isinstance(eigen_im_z1, torch.Tensor) else eigen_im_z1,
    eigen_im_single.cpu().numpy() if isinstance(eigen_im_single, torch.Tensor) else eigen_im_single], axis=0)
            input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(device)  # Shape: [1, 2, H, W]
            sigma = sigma_est
            if not x8 and input_tensor.size(2)//8==0 and input_tensor.size(3)//8==0:
                img_E = model(input_tensor)
            elif not x8 and (input_tensor.size(2)//8!=0 or input_tensor.size(3)//8!=0):
                img_E = utils_model.test_mode(model, input_tensor, refield=64, mode=5)
            elif x8:
                img_E = utils_model.test_mode(model, input_tensor, mode=3)
            if i == 0:
                eigen_im_z1 = img_E
            eigen_im_z1 = eigen_im_z1.squeeze()
            eigen_Y_eigencnn[i, :] = (img_E * scale + min_x).reshape(1, hight * width).cpu().numpy()
        # ------------------------------------
        # 加载adapter
        # ------------------------------------
        img_clean_dc_tensor = torch.from_numpy(img_clean_dc).float().to(device)
        model = SAH(k_subspace,k_subspace).cuda()
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas=(0.9, 0.999),eps=1e-8)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[250,350,450], gamma=0.5)  
        mse = torch.nn.MSELoss().cuda()
        U_tensor = torch.from_numpy(U).float().to(device)  # 将 U 转换为 PyTorch 张量并移动到设备
        Rw_ori_diag = torch.sqrt(torch.from_numpy(np.diag(Rw_ori)).float()).to(device)  # 提取 Rw_ori 的对角线并取平方根
        with torch.set_grad_enabled(True):
            for epoch in range(500):
                model.train()
                epoch_loss = 0
                optimizer.zero_grad()
                with torch.no_grad():
                    model_input = torch.from_numpy(eigen_Y_eigencnn).float().cuda().unsqueeze(0).reshape(1,k_subspace,hight,width)
                model_out = model(model_input)
                intermediate = torch.matmul(U_tensor, model_out.squeeze().reshape(k_subspace, hight * width))  # [L x N]
                target = (Rw_ori_diag.unsqueeze(1) * intermediate).transpose(0, 1).reshape(hight, width, channel)  # [H, W, C]
                #target = np.sqrt(Rw_ori).dot(U.dot(model_out.cpu().numpy().squeeze().reshape(k_subspace,hight*width))).T.reshape(hight, width, channel)
                loss = torch.sqrt(mse(target,img_clean_dc_tensor))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                if epoch % 1000 == 0:
                    print("epoch:{} loss:{}".format(epoch,epoch_loss))
        Y_reconst = U.dot(model_out.detach().cpu().numpy().squeeze().reshape(k_subspace,hight*width))
        Y_reconst = np.sqrt(Rw_ori).dot(Y_reconst)
        image_EigenCNN = Y_reconst.T.reshape(hight, width, channel)   

          

        
        psnr = util.calculate_psnr(image_EigenCNN, img_clean_dc, border=border)
        ssim = util.calculate_ssim(image_EigenCNN, img_clean_dc, border=border)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))

        # ------------------------------------
        # save results
        # ------------------------------------
        #保存img_E
        save_path = os.path.join(E_path, img_name + '_denoised.mat')
        scipy.io.savemat(save_path, {'img_E': img_E})
        logger.info('Saved denoised image to {:s}'.format(save_path))
        #util.imsave(img_E, os.path.join(E_path, img_name+ext))

    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    logger.info('Average PSNR/SSIM(RGB) - {} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, ave_psnr, ave_ssim))


if __name__ == '__main__':

    main()
