import os
import json
import torch
import numpy as np
import argparse
import torchvision
from PIL import Image
from surfree_utils.surfree import SurFree
from utils.JND_predict import JND
import cv2
import glob

from attack_LIVE import pil_loader,get_model_DBCNN,fix_seed,extra_edge,crop_totensor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", "-o", default="./outputs/LIVE_result_DBCNN_demo", help="Output folder")
    parser.add_argument("--loop_times", "-l", type=int, default=20, help="Loop times for attacking")
    parser.add_argument("--increase", "-incr", action='store_true', help="Increase or decrease strategy used")
    parser.add_argument("--ref_path", type=str, default='', help="Path to the ref image")
    parser.add_argument("--sal_path", type=str, default='./dataset/LIVE_crop_sal/fastfading/img106_original_gt67.1_MB.png', help="Path to the saliency image of the attacked image")
    parser.add_argument("--seed", type=int, default=919, help="Seed for fixed random")
    
    parser.add_argument(
        "--config_path", 
        default="utils/config_example.json", 
        help="Configuration Path with all the parameter for SurFree. It have to be a dict with the keys init and run."
        )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    fix_seed(args.seed)
    ###############################
    output_folder = args.output_folder
    print('output_folder',output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder,exist_ok=True)
        os.makedirs(os.path.join(output_folder,'fastfading'),exist_ok=True)
        os.makedirs(os.path.join(output_folder,'wn'),exist_ok=True)
        os.makedirs(os.path.join(output_folder,'jpeg'),exist_ok=True)
        os.makedirs(os.path.join(output_folder,'jp2k'),exist_ok=True)
        os.makedirs(os.path.join(output_folder,'gblur'),exist_ok=True)
        print("{} doesn't exist, makedir.".format(output_folder))

    print('Load DBCNN Model')
    from DBCNN.DBCNN_train_attack import DBCNN
    model = get_model_DBCNN()

    ###############################
    print("Load Config")
    if args.config_path is not None:
        if not os.path.exists(args.config_path):
            raise ValueError("{} doesn't exist.".format(args.config_path))
        config = json.load(open(args.config_path, "r"))
    else:
        config = {"init": {}, "run": {"epsilons": None}}

    ###############################
    print("Load Data")
    X = []
    X_JND = []
    transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(size=224),
                    torchvision.transforms.ToTensor(),])
    transforms2 = torchvision.transforms.Compose([
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
    transforms3 = torchvision.transforms.Compose([
                    torchvision.transforms.Normalize(mean=(0, 0, 0),
                                                     std=(0.229, 0.224, 0.225))])
    meana = torch.ones((3,224,224)).cuda()
    stda = torch.ones((3,224,224)).cuda()
    meana[0,:,:]=0.485
    meana[1,:,:]=0.456
    meana[2,:,:]=0.406
    stda[0,:,:]=0.229
    stda[1,:,:]=0.224
    stda[2,:,:]=0.225

    print('Using high frequency ref imgs.')
    ref_ims_list = ['./dataset/high_fre_imgs/I60.png', 
                    './dataset/high_fre_imgs/I71.png',] * 5
    print('Ref Images:',ref_ims_list)

    ref_ims = []
    for i,ref_im_path in enumerate(ref_ims_list):
        ref_im = pil_loader(ref_im_path)
        ref_im_np = np.array(ref_im)
        ref_im = transforms(ref_im)
        ref_im = transforms3(ref_im).cuda().unsqueeze(0)
        ref_ims.append(ref_im)
    
    increase = args.increase
    print('increase:',increase)

    saliency_dir = './dataset/LIVE_crop_sal'
    JND_thre = 0.38

    im_edges = []
    im_sals = []
    # imgpath = '/home/ycx/program/RobustIQA/attack_example/LIVE_crop_badpred_dbcnn/fastfading/img106_original_gt67.1.jpg'
    imgpath = '/home/ycx/program/RobustIQA/attack_example/LIVE_crop_goodpred_dbcnn/jp2k/img91_original_gt41.8.jpg'
    # imgpath = './dataset/LIVE_crop_goodpred_dbcnn/jp2k/img91_original_gt41.8.jpg'
    # imgpath = './dataset/LIVE_crop_badpred_dbcnn/fastfading/img106_original_gt67.1.jpg'
    im = pil_loader(imgpath)

    # Edge
    blur_edge = extra_edge(im)
    blur_edge = Image.fromarray(np.uint8(np.repeat(blur_edge[:,:,np.newaxis],3,axis=2)))
    

    # Saliency
    if args.sal_path != '':
        sal_path = args.sal_path
    else:
        im_subdir, im_name = imgpath.split('/')[-2:]
        sal_pathlist = glob.glob(os.path.join(saliency_dir, im_subdir,im_name[:-4]+'_MB+.png'))
        if len(sal_pathlist)>0:
            sal_path = sal_pathlist[0]
        else:
            raise ValueError('No saliency image provide')

    im_sal = pil_loader(sal_path)

    ims = crop_totensor([im,blur_edge,im_sal])
    im = ims[0]
    blur_edge = ims[1].unsqueeze(0).cuda()
    im_edges.append(blur_edge)
    im_sal = ims[2]
    im_sal = transforms3(im_sal).cuda()
    im_sals.append(im_sal.unsqueeze(0))

    # JND
    im_np = np.array(im*255).transpose(1,2,0)
    im_np = im_np.astype(np.uint8)
    im_gray=cv2.cvtColor(im_np,cv2.COLOR_RGB2GRAY)
    im_JND = JND_thre * JND(im_gray)
    im_JND = im_JND.unsqueeze(0).repeat(3,1,1)/255
    im_JND = transforms3(im_JND)
    X_JND.append(im_JND.unsqueeze(0))

    im = transforms2(im).unsqueeze(0).cuda()
    X.append(im)

    ref_ims = torch.cat(ref_ims, 0)
    im_edges = torch.cat(im_edges, 0)
    im_sals = torch.cat(im_sals, 0)

    X = torch.cat(X, 0)
    y = model(X).unsqueeze(0)
    X_JND = torch.cat(X_JND, 0)

    ###############################
    print("Attack!")
    if torch.cuda.is_available():
        model = model.cuda(0)
        X = X.cuda(0)
        y = y.cuda(0)
        X_JND = X_JND.cuda(0)

    loop_times = args.loop_times
    print('loop_times:',loop_times)
    start_X = X
    original_p = y
    init_amp= 1/100
    # print('Init amplitude for boundary:',init_amp)
    boundary=(torch.ones((X.shape[0]))*init_amp).cuda()
    last_boundary=torch.zeros((X.shape[0])).cuda()
    stop_attack=torch.tensor([False]*X.shape[0]).cuda()
    already_rand_t=0
    nqueries_list=[]
    for attack_time in range(loop_times):
        # print('Boundary:',boundary)
        f_attack = SurFree(**config["init"],boundary=boundary,last_boundary=last_boundary,increase=increase)
        advs, iter_boundary, stop_attack, already_rand_t = f_attack(model, start_X, y, X_ori=X, 
            ref=ref_ims, X_JND=X_JND, X_edges=im_edges, X_sals = im_sals, 
            stop_attack=stop_attack, already_rand_t=already_rand_t, **config["run"])
        if already_rand_t > 6000:
            print('Break in main: already_rand_t reaches maximum.')
            break
        if stop_attack.all():
            break

        start_X = advs
        # print('iter_boundary',iter_boundary)
        # print('last_boundary',last_boundary)
        boundary = iter_boundary + (iter_boundary-last_boundary)
        boundary[iter_boundary==0]=0
        last_boundary = iter_boundary.cuda()

    ###############################
    print("Results")
    query_times = []
    labels_advs = model(advs)
    MAEs = []
    dir,name = imgpath.split('/')[-2:]
    print("Adversarial Image {}/{}:".format(dir,name))
    print('labels_advs',labels_advs)
    print('y',y)
    label_adv = int(labels_advs.item())
    print("\t- Original label: {}".format(str(y[0].item())))
    print("\t- Adversarial label: {}".format(str(labels_advs.item())))
    print("\n")

    ###############################
    print("Save Results")
    meana = meana.cpu()
    stda = stda.cpu()
    o = X[0]
    o = o.cpu()*stda+meana
    o = np.array(o * 255).astype(np.uint8)
    dir,name = imgpath.split('/')[-2:]
    name = name.split('.')[0]

    img_o = Image.fromarray(o.transpose(1, 2, 0), mode="RGB")
    img_o.save(os.path.join(output_folder, dir, "{}_original_gt{}.jpg".format(name,str(y[0].item())[:4])))
    
    adv_i = advs[0].cpu()*stda+meana
    torch.save(adv_i,os.path.join(output_folder, dir, "{}_adversarial_JNDmask_b{:.3f}_l{}_p{}.pt".format(name,boundary[0],loop_times,str(labels_advs.item())[:4])))
    adv_i = np.array(adv_i * 255).astype(np.uint8)
    adv_i = Image.fromarray(adv_i.transpose(1, 2, 0), mode="RGB")
    adv_i.save(os.path.join(output_folder, dir, "{}_adversarial_gt{}.jpg".format(name,str(y[0].item())[:4])))    

    
# python attack_demo.py --incr