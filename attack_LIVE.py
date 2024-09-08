import os
import json
import torch
import torch.nn as nn
import numpy as np
import argparse
import torchvision
from PIL import Image
import random
from surfree_utils.surfree import SurFree
from utils.JND_predict import JND
import cv2
import h5py
import glob

def select_image(increase):
    ref_map = {'bikes.bmp': 1, 'building2.bmp': 2, 'buildings.bmp': 3, 'caps.bmp': 4, 
         'carnivaldolls.bmp': 5, 'cemetry.bmp': 6, 'churchandcapitol.bmp': 7, 'coinsinfountain.bmp': 8,
         'dancers.bmp': 9, 'flowersonih35.bmp': 10, 'house.bmp': 11, 'lighthouse.bmp': 12, 
         'lighthouse2.bmp': 13, 'manfishing.bmp': 14, 'monarch.bmp': 15, 'ocean.bmp': 16, 
         'paintedhouse.bmp': 17, 'parrots.bmp': 18, 'plane.bmp': 19, 'rapids.bmp': 20, 
         'sailing1.bmp': 21, 'sailing2.bmp': 22, 'sailing3.bmp': 23, 'sailing4.bmp': 24, 
         'statue.bmp': 25, 'stream.bmp': 26, 'studentsculpture.bmp': 27, 'woman.bmp': 28, 
         'womanhat.bmp': 29}

    exp_id=0
    INFO = h5py.File('./dataset/LIVEINFO.mat', 'r')
    ref_names = [INFO[INFO['ref_ids'][0, :][i]][()].tobytes()\
                        [::2].decode() for i in range(len(INFO['ref_ids'][0,:]))]
    dis_names = [INFO[INFO['im_names'][0, :][i]][()].tobytes()\
                    [::2].decode() for i in range(len(INFO['im_names'][0,:]))]
    mos = INFO['subjective_scores'][0, :]
    random_split = INFO['index'][:,exp_id]
    random_split = [int(i) for i in random_split]
    test_split=random_split[int(len(random_split)*0.8):]

    mos_list = []
    mos_dir_ori = {}
    for i in range(len(dis_names)):
        ref=ref_names[i]
        if int(ref_map[ref]) not in test_split:
            continue
        dis = dis_names[i]
        dis_name = dis.split('.')[0]
        mos_dir_ori[dis_name] = mos[i]

    if increase:
        im_dir = './dataset/LIVE_crop_goodpred_' + args.model
    else:
        im_dir = './dataset/LIVE_crop_badpred_' + args.model
    print('im_dir',im_dir)
    im_list = glob.glob(im_dir+"/*/*.jpg")
    print('len(im_list)',len(im_list))
    
    for i,impath in enumerate(im_list):
        imtype,imname = impath.split('/')[-2:]
        imname = imname.split('_')[0]
        mosi = mos_dir_ori[imtype+'/'+imname]
        mos_list.append(mosi)
    im_np = np.array(im_list)
    mos_np = np.array(mos_list)
    
    return im_np, mos_np

def get_model_DBCNN():
    from DBCNN.DBCNN_train_attack import DBCNN
    options = {'fc': True}
    scnn_root = './checkpoints/scnn.pkl'
    model = nn.DataParallel(DBCNN(scnn_root, options), device_ids=[0]).cuda()
    checkpoint_path = './checkpoints/LIVE_net_params_best.pkl'
    checkpoint = torch.load(checkpoint_path)
    print('Load from',checkpoint_path)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", "-o", default="./LIVE_result_DBCNN", help="Output folder")
    parser.add_argument("--n_images", "-n", type=int, default=8, help="N images attacks")
    parser.add_argument("--start", "-s", type=int, default=0, help="Start from which image")
    parser.add_argument("--loop_times", "-l", type=int, default=10, help="Loop times for attacking")
    parser.add_argument("--increase", "-incr", action='store_true', help="Increase or decrease strategy used")
    
    parser.add_argument("--model", "-m", type=str, default='dbcnn', help="Model to attack")
    parser.add_argument("--ref_path", type=str, default='None', help="Path to the ref image")

    parser.add_argument("--seed", type=int, default=919, help="Seed for fixed random")
    
    parser.add_argument(
        "--config_path", 
        default="utils/config_example.json", 
        help="Configuration Path with all the parameter for SurFree. It have to be a dict with the keys init and run."
        )
    return parser.parse_args()

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    np.random.seed(seed)
    random.seed(seed)
    print('Fix seed to ', seed)

def crop_totensor(imgs):
    # Input: list of narray H*W*C
    # Output: list of tensor 1*C*H*W
    crop_size = (224, 224)
    crop_position1 = random.randint(0,imgs[0].size[1]-crop_size[0])
    crop_position2 = random.randint(0,imgs[0].size[0]-crop_size[1])
    new_imgs = []
    for img in imgs:
        img = torchvision.transforms.functional.crop(img, top=crop_position1, left=crop_position2, height=crop_size[1], width=crop_size[0])
        img = torchvision.transforms.ToTensor()(img)
        new_imgs.append(img)
    return new_imgs

def recover01(im,mean,std):
    # tensor -> tensor range in [0,1]
    im2 = im*std+mean
    return im2
    
    
def extra_edge(img):
    img2 = np.array(img)
    edges = cv2.Canny(img2, 50, 150)
    blurred = cv2.GaussianBlur(edges, (3, 3), 0)
    return blurred

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

    ###############################
    if args.model == 'dbcnn':
        print('Load DBCNN Model')
        model = get_model_DBCNN()
    else:
        raise NotImplementedError

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

    ims_np,mos_np = select_image(increase)
    start = args.start
    print('Images:',ims_np[start*8:(start+1)*8])
    print('MOS:',mos_np[start*8:(start+1)*8])

    im_edges = []
    im_sals = []
    for i,img in enumerate(ims_np[start*8:(start+1)*8]): # 8 images are attacked at a time
        im = pil_loader(img)

        # Edge
        blur_edge = extra_edge(im)
        blur_edge = Image.fromarray(np.uint8(np.repeat(blur_edge[:,:,np.newaxis],3,axis=2)))
        ims = crop_totensor([im,blur_edge])
        im = ims[0]
        blur_edge = ims[1].unsqueeze(0).cuda()
        im_edges.append(blur_edge)

        # Saliency
        im_subdir, im_name = img.split('/')[-2:]
        sal_pathlist = glob.glob(os.path.join(saliency_dir, im_subdir,im_name[:-4]+'_MB+.png'))
        sal_path = sal_pathlist[0]
        im_sal = pil_loader(sal_path)
        im_sal= transforms(im_sal)
        im_sal= transforms3(im_sal).cuda()
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
    y = model(X)
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
    print('Init amplitude for boundary:',init_amp)
    boundary=(torch.ones((X.shape[0]))*init_amp).cuda()
    last_boundary=torch.zeros((X.shape[0])).cuda()
    stop_attack=torch.tensor([False]*X.shape[0]).cuda()
    already_rand_t=0
    count_adv_all=0
    count_adv_JND_all=0
    nqueries_list=[]
    for attack_time in range(loop_times):
        print('Boundary:',boundary)
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
        print('iter_boundary',iter_boundary)
        print('last_boundary',last_boundary)
        boundary = iter_boundary + (iter_boundary-last_boundary)
        boundary[iter_boundary==0]=0
        last_boundary = iter_boundary.cuda()


    ###############################
    print("Results")
    query_times = []
    labels_advs = model(advs)
    MAEs = []
    for image_i in range(len(X)):
        dir,name = ims_np[start*8+image_i].split('/')[-2:]
        print("Adversarial Image {}/{}:".format(dir,name))
        label_o = int(y[image_i])
        label_adv = int(labels_advs[image_i])
        print("\t- Original label: {}".format(str(y[image_i])))
        print("\t- Adversarial label: {}".format(str(labels_advs[image_i])))
        print("\n")
    ###############################
   
    print("Save Results")
    meana = meana.cpu()
    stda = stda.cpu()
    for image_i, o in enumerate(X):
        o = o.cpu()*stda+meana
        o = np.array(o * 255).astype(np.uint8)
        dir,name = ims_np[start*8+image_i].split('/')[-2:]
        name = name.split('.')[0]

        img_o = Image.fromarray(o.transpose(1, 2, 0), mode="RGB")
        img_o.save(os.path.join(output_folder, dir, "{}_original_gt{}.jpg".format(name,str(y[image_i].item())[:4])))
        
        adv_i = advs[image_i].cpu()*stda+meana
        torch.save(adv_i,os.path.join(output_folder, dir, "{}_adversarial_JNDmask_b{:.3f}_l{}_p{}.pt".format(name,boundary[image_i],loop_times,str(labels_advs[image_i].item())[:4])))
