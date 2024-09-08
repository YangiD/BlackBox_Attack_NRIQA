# from https://github.com/YogaLYJ/GMM-attack/blob/main/GMM.py
# reimplement from Just Noticeable Difference for Images with Decomposition Model for Separating Edge and Textured Regions
import torch
import cv2
import torch.nn.functional as F
import numpy as np
import os

def fix_seed(seed):
    torch.manual_seed(seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    np.random.seed(seed)

def LA(img):
    m, n = img.shape
    LA_mask = np.zeros((m, n))
    save_list = np.zeros(256)
    for i in range(128):
        save_list[i] = 17 * (1 - np.sqrt(i / 127)) + 3
    for i in range(128,256):
        save_list[i] = 3 * (i - 127) / 128 + 3

    for i in range(m):
        for j in range(n):
            LA_mask[i][j] = save_list[img[i][j]]
        
 
    return LA_mask

def LA_th(img):
    m, n = img.shape
    LA_mask = torch.zeros((m, n)).cuda()
    save_list = torch.zeros(256).cuda()
    for i in range(128):
        save_list[i] = 17 * (1 - np.sqrt(i / 127)) + 3
    for i in range(128,256):
        save_list[i] = 3 * (i - 127) / 128 + 3

    for i in range(m):
        for j in range(n):
            LA_mask[i][j] = save_list[img[i][j].long()] 

    return LA_mask

def CM(img):
    structural_img = cv2.Canny(img, 50, 150)
    max_img = F.max_pool2d(input=torch.Tensor(structural_img).unsqueeze(0), kernel_size=5, padding=2, stride=1)
    min_img = -F.max_pool2d(input=-torch.Tensor(structural_img).unsqueeze(0), kernel_size=5, padding=2, stride=1)
    C = (max_img - min_img).squeeze(0)
    EM_mask = C * 0.117

    textural_img = img - structural_img
    max_img = F.max_pool2d(input=torch.Tensor(textural_img).unsqueeze(0), kernel_size=5, padding=2, stride=1)
    min_img = -F.max_pool2d(input=-torch.Tensor(textural_img).unsqueeze(0), kernel_size=5, padding=2, stride=1)
    C = (max_img - min_img).squeeze(0)
    TM_mask = C * 0.117 * 3

    CM_mask = EM_mask + TM_mask

    return CM_mask


# Compute JND mask
def JND(img):
    # narray H*W -> tensor H*W
    LA_img = torch.Tensor(LA(img))
    # # tensor H*W -> tensor H*W
    # LA_img = LA_th(img)
    CM_img = CM(img)
    JND_mask = LA_img + CM_img - 0.3*torch.min(LA_img,CM_img)

    return JND_mask

def add_noise(img,mask):
    mask = mask/255
    mean = 0
    var = 3
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    noise = np.clip(noise,-8,8)
    p = np.random.rand(img.shape[0],img.shape[1],img.shape[2])
    mask = mask[:,:,np.newaxis]
    img_new = img+noise
    img_new = np.clip(img_new,0,255)
    return img_new

if __name__ == '__main__':
    fix_seed(919)
    savedir = 'YOUR_SAVE_DIR'
    imgpath='YOUR_TEST_IMG'
    imgname = imgpath.split('/')[-1]
    img=cv2.imread(imgpath)
    img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img_JND=JND(img_gray)
    img_JND=np.array(img_JND)
    img_noise=add_noise(img,img_JND)
    change_mask = np.abs(img_noise-img)>0
    change_mask = np.ones_like(change_mask)*change_mask
    savepath = os.path.join(savedir,imgname)
    cv2.imwrite(os.path.join(savedir,imgname),img_noise)

