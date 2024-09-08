import glob
import torch
import torchvision
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau
from attack_LIVE0 import pil_loader,get_model_DBCNN

def recover01(im,mean,std):
    # tensor -> tensor range in [0,1]
    im2 = (im*std+mean) #*255
    return im2
    
def score_normalize(score):
    mos_min,mos_max = 3.42, 92.43195
    score = (score-mos_min)/(mos_max-mos_min)*100
    score = 100-score
    return score

if __name__ == "__main__":
    modelname = 'dbcnn'
    dir2 = True

    test_dir = './outputs/LIVE_result_DBCNN_incr_l20_sd919'
    print('test_dir',test_dir) 
    img_list = glob.glob(os.path.join(test_dir,"*/*.[jp][pn][g]"))
    img_list15 = glob.glob(os.path.join(test_dir,"*/*.pt"))
    img_list = img_list + img_list15
    test_dir2 = './outputs/LIVE_result_DBCNN_decr_l20_sd919'
    print('test_dir2',test_dir2)
    img_list2 = glob.glob(os.path.join(test_dir2,"*/*.[jp][pn][g]"))
    img_list25 = glob.glob(os.path.join(test_dir2,"*/*.pt"))
    img_list2 = img_list2 + img_list25
    img_list = img_list + img_list2


    print('len(img_list)',len(img_list))# 320
    
    mos_dir = np.load('./dataset/mos_dir.npy',allow_pickle=True).item()

    if modelname == 'dbcnn':
        print('Load DBCNN Model')
        model = get_model_DBCNN()
    else:
        raise NotImplementedError
    
    print("Load Data")
    y = []
    transforms = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
    transforms_norm = torchvision.transforms.Compose([
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
    for i,img in enumerate(img_list):
        if img.endswith(('png','jpg')):
            im = pil_loader(img)
            im = transforms(im).unsqueeze(0).cuda()
        else:
            im = torch.load(img)
            im = transforms_norm(im).unsqueeze(0).cuda()
        yi = model(im)
        y.append(float(yi.item()))

    
    ori_pred = []
    ori_mos = []
    attack_pred = []
    attack_mos = []
    # check_dir = {imname:0 for imname in mos_dir}
    score_dir = {}

    for i,impath in enumerate(img_list):
        dir, imname = impath.split('/')[-2:]
        if '_diff' in impath:
            continue

        imnum = imname.split('_')[0]
        if len(imname.split('_'))>3:
            type = imname.split('_')[3]
        elif len(imname.split('_'))==3: ## for UAP
            type = 'original'
        elif len(imname.split('_'))<2:
                print('len<2',imname)
                os._exit(0)

        if type=='adversarial':
            pred_befo = imname.split('_')[-1][1:-4]
        
        dir_im = os.path.join(dir, imnum)


        if dir_im not in mos_dir:
            print('Not exist:',dir_im)
            continue
        if dir_im not in score_dir:
            score_dir[dir_im] = [0,0,0]
        if type=='original':
            score_dir[dir_im][0] = mos_dir[dir_im]
            score_dir[dir_im][1] = y[i]

        elif type=='adversarial':
            score_dir[dir_im][2] = y[i]

        # check_dir[dir_im]=1

    im_names = []
    for imname in score_dir:
        scorelist = score_dir[imname]
        ori_mos.append(scorelist[0])
        ori_pred.append(scorelist[1])
        attack_pred.append(scorelist[2])
        

    ori_pred = np.array(ori_pred).squeeze()
    ori_mos = np.array(ori_mos).squeeze()
    attack_pred = np.array(attack_pred).squeeze() 
    ori_diff = np.mean(np.abs(ori_pred-ori_mos))
    attack_diff = np.mean(np.abs(attack_pred-ori_mos))
    

    rho_s, _ = spearmanr(ori_mos, ori_pred)
    rho_p, _ = pearsonr(ori_mos, ori_pred)
    rho_k, _ = kendalltau(ori_mos, ori_pred)
    print(' Original: {0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4}'.format(rho_s,rho_p,rho_k,ori_diff))
    rho_s, _ = spearmanr(ori_mos, attack_pred)
    rho_p, _ = pearsonr(ori_mos, attack_pred)
    rho_k, _ = kendalltau(ori_mos, attack_pred)
    print(' Attack: {0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4}'.format(rho_s,rho_p,rho_k,attack_diff))
    rho_s1, _ = spearmanr(ori_pred, attack_pred)
    rho_p1, _ = pearsonr(ori_pred, attack_pred)
    rho_k1, _ = kendalltau(ori_pred, attack_pred)
    badiff = np.mean(np.abs(attack_pred-ori_pred))
    print(' Attack, Before & After Attack: \n{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4}'.format(rho_s,rho_p,rho_k,attack_diff,rho_s1,rho_p1,rho_k1,badiff))