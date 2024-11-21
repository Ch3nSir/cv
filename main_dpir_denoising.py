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
from utils import utils_pnp as pnp
from Hysime import *
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
    L = 8                               # L is the number of selected columns
    iter_num = 10                         # default: 10, 20 or 30 for different noise levels
    modelSigma1 = 49                     # set sigma_1, default: 49
    modelSigma2 = noise_level_model*255. # set sigma_2, default: noise_level_model*255

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
        Noise_L_image = img_L
        #svd
        Y = np.transpose(Noise_L_image, (2, 0, 1)).reshape([channel, hight * width])
        #将三维数组 x 转置并重塑为二维数组 Y，形状为 [channel, hight*width]。 
        U1,sigma,Vt = la.svd(Y,full_matrices=False)
        #对 u_t 进行奇异值分解，得到左奇异向量矩阵 U1、奇异值向量 sigma 和右奇异向量矩阵的转置 Vt。
        U = U1[:, 0:L]                        #U是光谱表示，L从列里选取，可调 U(191,p)
        # --------------------------------
        # (3) get rhos and sigmas
        # --------------------------------
        rhos, sigmas = pnp.get_rho_sigma(sigma=max(0.255/255., noise_level_img), iter_num=iter_num, modelSigma1=modelSigma1, modelSigma2=modelSigma2, w=1.0)
        #rhos, sigmas = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device)
        # ------------------------------------
        # (2) img_E
        # ------------------------------------
        img_E = np.matmul(U.T,Y)               #E=Ut*Y 计算空间表示系数 E，通过将 Y 投影到光谱基 U 上。 img_E(p,hight*width)
            #if not x8 and E.size(2)//8==0 and E.size(3)//8==0:
            #    img_E = model(E)
            #elif not x8 and (E.size(2)//8!=0 or E.size(3)//8!=0):
            #    img_E = utils_model.test_mode(model, E, refield=64, mode=5)
            #elif x8:
            #   img_E = utils_model.test_mode(model, E, mode=3)
        for i in range(iter_num):
            # --------------------------------
            # step 1, 递归下降法更新更新x
            # --------------------------------
            img_E = img_E - rhos[i] * (np.matmul(U.T, np.matmul(U, img_E) - Y))  #更新 img_E(p,hight*width)

            # --------------------------------
            # step 2, denoiser
            # --------------------------------
            for j in range(L):               
                E = img_E[j,:]               #E=Ut*Y 计算空间表示系数 E，通过将 Y 投影到光谱基 U 上。 E(hight*width)
                E = np.expand_dims(E, axis=1) #将 E 调整为二维数组，以便输入网络。
                E_noisy,Rw = est_noise(E)      #获取噪声估计  E_noisy和噪声相关矩阵 Rw。
                E = E.reshape(1,hight,width) #将 E 重塑回三维数组，并调整维度顺序。
                util.imshow(util.single2uint(E), title='Noisy image with noise level {}'.format(noise_level_img)) if show_img else None
                E_noisy = E_noisy.reshape(1,hight,width).transpose(1, 2, 0) #将 E_noisy 重塑回三维数组，并调整维度顺序。
                E_noisy = util.single2tensor4(E_noisy)
                E = util.single2tensor4(E)
                #img_L = torch.cat((img_L, torch.FloatTensor([noise_level_model/255.]).repeat(1, 1, img_L.shape[2], img_L.shape[3])), dim=1)
                E = E.permute(0, 2, 3, 1)
                E = torch.cat((E, E_noisy.reshape(1,1, hight, width)), dim=1)
                E = E.to(device)

                if not x8 and E.size(2)//8==0 and E.size(3)//8==0:
                    E = model(E)
                elif not x8 and (E.size(2)//8!=0 or E.size(3)//8!=0):
                    E = utils_model.test_mode(model, E, refield=64, mode=5)
                elif x8:
                    E = utils_model.test_mode(model, E, mode=3)
            # ------------------------------------
            #重新拼接E
            # ------------------------------------
                img_cat = torch.cat((img_cat, E), dim=0) if j > 0 else E
                print(img_cat.shape)
            img_E = img_cat
            print(img_E.shape)
            img_E = img_E.reshape(L,hight*width)
            img_E = util.tensor2uint(img_E)
            print(img_E.shape)

            # --------------------------------
            # PSNR and SSIM
            # --------------------------------
        img_E = np.matmul(U,img_E) 
        img_E = img_E.reshape(hight,width,channel)
        psnr = util.calculate_psnr(img_E, img_clean_dc, border=border)
        ssim = util.calculate_ssim(img_E, img_clean_dc, border=border)
        print(psnr)
        print(ssim)
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
