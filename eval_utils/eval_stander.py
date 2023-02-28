import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
import cv2
import numpy as np
import torch
import lpips

loss_fn = lpips.LPIPS(net='vgg').to('cuda')

# # 加载图像并转换为张量
# img1 = cv2.imread('path/to/image1.jpg')
# img1 = np.transpose(img1, (2, 0, 1)) / 255.0
# img1 = torch.from_numpy(img1).unsqueeze(0).float().to('cuda')

# img2 = cv2.imread('path/to/image2.jpg')
# img2 = np.transpose(img2, (2, 0, 1)) / 255.0
# img2 = torch.from_numpy(img2).unsqueeze(0).float().to('cuda')

# # 计算 LPIPS
# lpips_score = loss_fn(img1, img2).item()




filePath = '/home/yiran/szx/volrend/output_img/temp/'
filelist=os.listdir(filePath)
psnr_value_co=0.0
ssim_value_co=0.0
lpips_score_co=0.0

psnr_value_ro=0.0
ssim_value_ro=0.0
lpips_score_ro=0.0

psnr_value_trans=0.0
ssim_value_trans=0.0
lpips_score_trans=0.0

psnr_value_plen=0.0
ssim_value_plen=0.0
lpips_score_plen=0.0

# 读取两张图片
for filename in filelist:
    imgco = cv2.imread("/home/yiran/szx/volrend/output_img/copy/"+ filename)
    imgro = cv2.imread("/home/yiran/szx/volrend/output_img/rotate/"+ filename)
    imgtrans = cv2.imread("/home/yiran/szx/volrend/output_img/trans/"+ filename)
    imgplen = cv2.imread("/home/yiran/szx/volrend/output_img/plenoctree/"+ filename)
    imggt = cv2.imread("/home/yiran/szx/nerf_synthetic/ship/test/"+filename,cv2.IMREAD_UNCHANGED)
    B, G, R, A = cv2.split(imggt)
    alpha = A / 255
    R = (255 * (1 - alpha) + R * alpha).astype(np.uint8)
    G = (255 * (1 - alpha) + G * alpha).astype(np.uint8)
    B = (255 * (1 - alpha) + B * alpha).astype(np.uint8)
    imggt = cv2.merge((B, G, R))
    
    


    gray_imgco = cv2.cvtColor(imgco, cv2.COLOR_BGR2GRAY)
    gray_imgro = cv2.cvtColor(imgro, cv2.COLOR_BGR2GRAY)
    gray_imgtrans = cv2.cvtColor(imgtrans, cv2.COLOR_BGR2GRAY)
    gray_imgplen = cv2.cvtColor(imgplen, cv2.COLOR_BGR2GRAY)
    gray_imggt = cv2.cvtColor(imggt, cv2.COLOR_BGR2GRAY)
    
    imgco = np.transpose(imgco, (2, 0, 1)) / 255.0
    imgco = torch.from_numpy(imgco).unsqueeze(0).float().to('cuda')
    
    imgro = np.transpose(imgro, (2, 0, 1)) / 255.0
    imgro = torch.from_numpy(imgro).unsqueeze(0).float().to('cuda')
    
    imgtrans = np.transpose(imgtrans, (2, 0, 1)) / 255.0
    imgtrans = torch.from_numpy(imgtrans).unsqueeze(0).float().to('cuda')
    
    imgplen = np.transpose(imgplen, (2, 0, 1)) / 255.0
    imgplen = torch.from_numpy(imgplen).unsqueeze(0).float().to('cuda')
    
    imggt = np.transpose(imggt, (2, 0, 1)) / 255.0
    imggt = torch.from_numpy(imggt).unsqueeze(0).float().to('cuda')

    psnr_value_co += psnr(gray_imgco, gray_imggt)
    ssim_value_co += ssim(gray_imgco, gray_imggt)
    lpips_score_co += loss_fn(imgco, imggt).item()
    
    psnr_value_ro += psnr(gray_imgro, gray_imggt)
    ssim_value_ro += ssim(gray_imgro, gray_imggt)
    lpips_score_ro += loss_fn(imgro, imggt).item()
    
    psnr_value_trans += psnr(gray_imgtrans, gray_imggt)
    ssim_value_trans += ssim(gray_imgtrans, gray_imggt)
    lpips_score_trans += loss_fn(imgtrans, imggt).item()
    
    psnr_value_plen += psnr(gray_imgplen, gray_imggt)
    ssim_value_plen += ssim(gray_imgplen, gray_imggt)
    lpips_score_plen += loss_fn(imgplen, imggt).item()


print("co PSNR avg:", psnr_value_co/len(filelist))
print("co SSIM avg:", ssim_value_co/len(filelist))
print("co LPIPS avg:", lpips_score_co/len(filelist))
print("\n")

print("ro PSNR avg:", psnr_value_ro/len(filelist))
print("ro SSIM avg:", ssim_value_ro/len(filelist))
print("ro LPIPS avg:", lpips_score_ro/len(filelist))
print("\n")

print("trans PSNR avg:", psnr_value_trans/len(filelist))
print("trans SSIM avg:", ssim_value_trans/len(filelist))
print("trans LPIPS avg:", lpips_score_trans/len(filelist))
print("\n")

print("plen PSNR avg:", psnr_value_plen/len(filelist))
print("plen SSIM avg:", ssim_value_plen/len(filelist))
print("plen LPIPS avg:", lpips_score_plen/len(filelist))



