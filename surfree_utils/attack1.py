import torch
# from utils.utils import atleast_kdim
from surfree_utils.utils import atleast_kdim
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os 
import random

def color_match(imgs,ref_imgs):
    # 将颜色通道中的中间值匹配一下
    # input:tensor BCHW input:tensor BCHW
    # 对图像排名
    new_ref_imgs = []
    for i in range(imgs.shape[0]):
        img = imgs[i] #([3, 2, 2])
        ref = ref_imgs[i]
        img_sum = img.sum(axis=0) #([2, 2])
        img_percent = img/img_sum #([3, 2, 2])
        img_percent_mean = img_percent.flatten(1).mean(dim=1)  #torch.Size([3]) median(.values)
        # print(img_percent_median) tensor([0.3333, 0.6667, 0.0000])
        img_rank = torch.sort(img_percent_mean,descending=True)
        # print(img_rank)
        # torch.return_types.sort(
        # values=tensor([0.6667, 0.3333, 0.0000]),
        # indices=tensor([1, 0, 2]))
        ref_percent_mean = ref.flatten(1).mean(dim=1) #torch.Size([3]) median .values
        ref_rank = torch.sort(ref_percent_mean,descending=True)
        ref = ref[ref_rank.indices]
        new_ref = ref[img_rank.indices]
        new_ref_imgs.append(new_ref.unsqueeze(0))
    new_ref_imgs = torch.cat(new_ref_imgs,0)
    return new_ref_imgs


def get_init_with_noise(model, X, y, X_ori=None, max_MOS=None, ref=None, \
                        boundary=None, last_boundary=None, X_JND=None, X_edges=None,  \
                        X_sals = None, stop_attack=None, increase=False, already_rand_t=None): #, original_p=None
    # already_rand_t: 在get_init_with_noise中，已经搜索了多少轮的random search
    max_queries = 200
    print('max_queries in get_init_with_noise:',max_queries)
    # boundary可变
    adptive_boundary = True # False #
    if adptive_boundary:
        print('Using Adaptive Boundary')
    else:
        print('Using Fixed Boundary')
    init = X.clone()# (a,b) torch.Size([2, 3, 224, 224])
    # print('y',y) # shape:(B)
    # stop_attack = torch.tensor([False]*X.shape[0])

    # # 观察100次里，被攻击成功的次数
    # counts = 0
    # for i in range(100):
    #     init = (X + 0.15*torch.randn_like(X)).clip(0, 1)
    #     # torch.where(
    #     #     atleast_kdim(p == y, len(X.shape)), 
    #     #     (X + 0.5*torch.randn_like(X)).clip(0, 1), 
    #     #     init)
    #     p = model(init) #.argmax(1)
    #     count = (p-y).sum()
    #     # print('count',count) #tensor(2)
    #     # print('count',count.shape) # torch.Size([])
    #     counts += torch.abs(count).detach().cpu()
    # print('counts',counts) # 0.5: 9548.7832 0.1: 493.3960 0.2: 1875.3552 0.15:1038.0006
    # os._exit(0)

    # perturb_already = X - X_ori
    # print('perturb_already.min()0',perturb_already.min())
    # print('perturb_already.max()0',perturb_already.max())
    trans_in_plus_minus = True
    if trans_in_plus_minus:
        print('Using coefficients of ref_im ranging from -1 to 1')

    if increase:
        use_color_match = False # True #
    else:
        use_color_match = False
    # use_color_match = True #For CLIVE!
    # print('For CLIVE:')

    if use_color_match:
        print('Using color match in ref_im')
    else:
        print('No color match used')

    if max_MOS is not None:
        # p = original_p if original_p is not None else model(X) ###
        p = model(X) ###  torch.Size([8])
        # print('max_MOS.shape',max_MOS.shape)
        if increase:
            condiction = (p <= y+(abs(max_MOS-y))*boundary) #(y+max_MOS)/2 #y+(max_MOS-y)/100 /40
        else:
            condiction = (p >= y-(abs(max_MOS-y))*boundary)
        condiction = condiction * (~stop_attack) #stop_attack为True的地方，cond改为False，用来不再检验
        # print('condiction',condiction) #torch.Size([8])
        stop_attack1 = torch.tensor([False]*X.shape[0]).cuda() 

        # 更新JND_mask
        perturb_already = X - X_ori
        perturb_already_minus = torch.where(perturb_already<0,perturb_already,0)
        perturb_already_plus = torch.where(perturb_already>0,perturb_already,0)
        # X_JND_rest = X_JND - perturb_already
        X_JND_minus_rest = -X_JND - perturb_already_minus
        X_JND_plus_rest = X_JND - perturb_already_plus
        # X_testJND = X+X_JND_plus_rest
        # X_testJND2 = X+X_JND_minus_rest
        # print('for X_JND_plus_rest')
        # for i in range(len(X)):
        #     imgplusi = X_testJND[i]-X_ori[i]
        #     imgminusi = X_testJND2[i]-X_ori[i]
        #     plusi = X_JND[i]-imgplusi
        #     print('plusi.min()',plusi.min()) #看看有没有负的 第一次就有 有问题-> 很小 可以理解为数值不稳定
        #     minusi = imgminusi-(-X_JND[i])
        #     print('minusi.min()',minusi.min()) #看看有没有负的 第一次就有 有问题
        #     # diffi = X_JND[i]-abs(imgi)
        #     diffi = X_JND[i]-X_JND_plus_rest[i]
        #     print('diffi.max()',diffi.max()) #先0，后面增大
        #     print('diffi.min()',diffi.min())

        meana = torch.ones_like(X).cuda() #torch.Size([2, 3, 224, 224])
        stda = torch.ones_like(X).cuda()
        meana[:,0,:,:]=0.485
        meana[:,1,:,:]=0.456
        meana[:,2,:,:]=0.406
        stda[:,0,:,:]=0.229
        stda[:,1,:,:]=0.224
        stda[:,2,:,:]=0.225

    else:
        p = model(X).argmax(1) #返回指定维度中最大的 #(a,1)
        condiction = (p == y)
    if ref is not None: #(1,3,224,224)
        print('ref in get_init_with_noise.')
        if ref.shape[0]==1:
            ref = ref.repeat(X.shape[0],1,1,1) #->([3, 3, 224, 224])
            random_ref = False
            print('One ref in get_init_with_noise.')
        elif ref.shape[0]==X.shape[0]:
            random_ref = False
            print('Specific ref of each one in get_init_with_noise.')
        else:
            random_ref = True
            scale = 0.1
            print('{} ref in get_init_with_noise.'.format(len(ref)))
            print('scale:', scale)
    else:
        random_ref = False
        print('Using 0.5*Gnoise in get_init_with_noise.')
        print('Using 1 or -1 in row in get_init_with_noise.')
    if X_edges is not None:
        print('Using X_edges in get_init_with_noise.')
    if X_sals is not None:
        print('Using X_sals in get_init_with_noise.')

    random_times = 0
    while any(condiction): #p == y #若有一个True则True，否则False #torch.Size([8])
        ### 减小boundary的机制 ###
        random_times+=1
        if random_times>max_queries: # 减小难以搜到x_b的boundary
            if adptive_boundary:
                stop_attack1 = condiction * ((boundary-last_boundary)<=1/400) #对boundary<=1/200，且200步没有找到的，标记停止计算
                print('stop_attack1',stop_attack1)
                if stop_attack1.any(): # 即当前已经达到了最小的boundary，但搜了max_queries次仍然搜不到
                    print('One case cannot find x_b, boundary:',boundary)
                    condiction = condiction * (~stop_attack1)# stop_attack里是true的地方，在condition中标为fasle
                    if not any(condiction): #如果只剩这一个有问题的了
                        print('Break.')
                        break
                # 若减小后,该iter的boundary（即(boundary-last_boundary)）>1/200,则减小；否则标记停止
                inf_boundary = last_boundary + (boundary-last_boundary)/2
                # print('boundary',boundary) #检查一下大小关系 没问题
                # print('inf_boundary',inf_boundary)
                desc_mask = condiction * ((boundary-last_boundary)>1/400) # 减小boundary的标准：max_queries次没有找到x_b,且当前boundary>1/200
                boundary = boundary + desc_mask*(inf_boundary-boundary) # boundary减小
                # print('condiction',condiction)
                # print('desc_mask',desc_mask)
                print('One boundary decreased case, changed boundary:',boundary)
                random_times = 1
                already_rand_t += max_queries
                print('already_rand_t:',already_rand_t)
                if already_rand_t > 20 * 300: # 20在这里是l max_queries # 给比较大的容许
                    print('Break.: already_rand_t reaches maximum.')
                    stop_attack1 = condiction
                    condiction = torch.tensor([True]*condiction.shape[0]).cuda()
                    break
            else: #固定边界
                stop_attack1 = condiction #torch.tensor([True]*condiction.shape[0]).cuda()
                condiction = torch.tensor([True]*condiction.shape[0]).cuda()
                print('One stop for fixed boundary.')
                break
        #######
            
        if max_MOS is not None:
            randi = torch.rand(X.shape[0]).unsqueeze(1).unsqueeze(1).unsqueeze(1).cuda() #->([3, 1, 1, 1])
            if trans_in_plus_minus:
                randi = (randi-0.5)*2
            if random_ref:
                ref_choice = random.choice(range(len(ref)))
                rand_ref = scale * randi * ref[ref_choice].repeat(X.shape[0],1,1,1) #
                region_mask = None
                if X_edges is not None:
                    region_mask = torch.where(X_edges>0,1,0)
                if X_sals is not None:
                    region_mask2 = torch.where(X_sals>1,1,0)
                    if X_edges is not None:
                        region_mask = torch.where(region_mask+region_mask2>0.5,1,0)
                    else:
                        region_mask = region_mask2
                # if X_edges is not None:   
                #     rand_ref = torch.where(X_edges>0,rand_ref,0)
                if region_mask is not None:
                    rand_ref = torch.where(region_mask>0,rand_ref,0)
            else:
                if ref is not None:
                    rand_ref = 0.5*randi*ref #([3, 3, 224, 224]) #0.2 0.5
                else:
                    # 全部初始化
                    # rand_ref = 0.5*torch.rand_like(X) 
                    # 条状初始化
                    B,C,H,W = X.shape
                    rand_ref = torch.randint(0, 2, (B,C,1,W))-1
                    rand_ref = (rand_ref*2/255).float().cuda()
                    rand_ref = (rand_ref)/stda
                    # rand_ref = torch.einsum('bc1w'->'bchw')
                    
                # rand_mask = torch.rand_like(X)
                # rand_mask = torch.where(rand_mask>0.5,1,-1)
                # rand_mask = rand_mask*5/255
                # rand_mask = (rand_mask)/stda # - meana
                # rand_ref = torch.where(ref>0, rand_mask,0)

            if use_color_match:
                rand_ref = color_match(X,rand_ref).cuda() #torch.Size([8, 3, 224, 224]) -> torch.Size([8, 3, 224, 224])
                
            # # 试一下使用（-1，1）的噪声
            # rand_mask = torch.rand_like(X)
            # rand_mask = torch.where(rand_mask>0.5,1,-1)
            # rand_mask = rand_mask*5/255
            # rand_mask = (rand_mask - meana)/stda
            # rand_ref = rand_mask

            # ref_JND = torch.where(X_JND_rest>rand_ref, rand_ref, X_JND_rest) #截断，大于 #X_JND
            rand_ref_plus = torch.where(rand_ref>0,rand_ref,0)
            rand_ref_minus = torch.where(rand_ref<0,rand_ref,0)
            ref_JND_plus = torch.where(X_JND_plus_rest>rand_ref_plus, rand_ref_plus, X_JND_plus_rest) #截断，大于 #X_JND
            ref_JND_minus = torch.where(X_JND_minus_rest<rand_ref_minus, rand_ref_minus, X_JND_minus_rest) #截断，小于JND下界的，截断为下界 #X_JND
            ref_JND = ref_JND_plus + ref_JND_minus
            # print('(X_JND-abs(ref_JND)).min()',(X_JND-abs(ref_JND)).min()) #有问题 一直都是负的
            # print('(X_JND-abs(X_JND_plus_rest)).min()',(X_JND-abs(X_JND_plus_rest)).min()) # 没问题
            # print('(X_JND-abs(X_JND_minus_rest)).min()',(X_JND-abs(X_JND_minus_rest)).min()) # 没问题
            
            X1=(X + ref_JND).clip(-meana/stda, (1-meana)/stda)

            # print('For X1')
            # for i,booli in enumerate(condiction): # 没问题，在范围内
            #     if not booli:
            #         imgi = X1[i]-X_ori[i]
            #         diffi = X_JND[i] - abs(imgi) # 有超过JND的地方
            #         print('diffi.min()',diffi.min())
            #         print('X_JND.max()',X_JND[i].max())
            #         print('imgi.max()',imgi.max())
            #         print('imgi.min()',imgi.min())
            # print('ref_JND[0].max()',ref_JND[0].max())
            # print('X_JND_plus_rest[0].max()',X_JND_plus_rest[0].max())
            # print('ref_JND[0].min()',ref_JND[0].min())
            # print('X_JND_minus_rest[0].min()',X_JND_minus_rest[0].min())
            # print('ref_JND[1].max()',ref_JND[1].max())
            # print('X_JND_plus_rest[1].max()',X_JND_plus_rest[1].max())
            # print('ref_JND[1].min()',ref_JND[1].min())
            # print('X_JND_minus_rest[1].min()',X_JND_minus_rest[1].min())

            init = torch.where(
                atleast_kdim(condiction, len(X.shape)), # p == y # 用于维持X的形状 
                X1, #(X + ref_JND).clip(-meana/stda, (1-meana)/stda), #0.5 (4/255)*ref torch.randn_like(X) 0.2*randi*ref
                init)
        else:
            init = torch.where(
                atleast_kdim(condiction, len(X.shape)), # p == y # 用于维持X的形状 
                (X + (4/255)*torch.randn_like(X)).clip(0, 1), #0.5
                init)
            
        if max_MOS is not None:
            p = model(init)
            ## 检验是否需要增加boundary ###
            if adptive_boundary:
                # sup_boundary = torch.max(boundary + (boundary-last_boundary)*2,torch.tensor(0.005))
                sup_boundary = torch.max(boundary + (boundary-last_boundary)/2,torch.tensor(0.005))
                if increase:
                    incr_mask = p > y+(abs(max_MOS-y))*sup_boundary #增加boundary的条件：得到的分数大于两倍的boundary
                    
                else:
                    incr_mask = p < y-(abs(max_MOS-y))*sup_boundary #增加boundary的条件：得到的分数小于两倍的boundary
                # incr_mask = torch.where(boundary>=0.8,False,incr_mask) # boundary>=0.4的地方，停止增加(incr_mask=False)

                if incr_mask.any(): # 有任何一个样本满足增加条件，则更改boundary
                    # print('p',p)
                    # print('y-(abs(max_MOS-y))*sup_boundary',y-(abs(max_MOS-y))*sup_boundary)
                    # print('incr_mask',incr_mask)
                    real_boundary = abs(p-y)/abs(max_MOS-y)-0.005 # for incr and decr     
                    print('sup_boundary',sup_boundary)
                    print('real_boundary',real_boundary)
                    print('boundary0',boundary)
                    # boundary = boundary + incr_mask*(sup_boundary-boundary)/2 #boundary -> 1.5*boundary
                    boundary = incr_mask*real_boundary
                    print('boundary1',boundary) # boundary1和boundary0应该是不一样的
                    print('One boundary increased case, changed boundary:',boundary)
            #######
            if increase:
                condiction = (p <= y+(abs(max_MOS-y))*boundary) #(y+max_MOS)/2  #y+(max_MOS-y)/100 /40  
            else:
                condiction = (p >= y-(abs(max_MOS-y))*boundary)
            condiction = condiction * (~stop_attack) #torch.Size([8])
            # print('condition',condiction)
        else:
            p = model(init).argmax(1)
            condiction = (p == y)
        
        ### 更新一下X ###
        print('p',p)
        print('random_times',random_times)
    already_rand_t += random_times

    print('for attack output')
    for i,booli in enumerate(condiction): # 没问题，在范围内
        if not booli:
            imgi = init[i]-X_ori[i]
            diffi = X_JND[i] - abs(imgi)
            # print('X_JND_plus_rest[i].max()',X_JND_plus_rest[i].max()) 
            # print('X_JND_minus_rest[i].min()',X_JND_minus_rest[i].min())
            print('X_JND.max()',X_JND[i].max())
            print('imgi.max()',imgi.max()) #第二次就有问题了
            print('diffi.min()',diffi.min()) #第一次就有问题了
            print('imgi.min()',imgi.min())
            
    # print('init-X',init-X)
    # for i,booli in enumerate(condiction):
    #     if not booli:
    #         # print('booli',booli)
    #         imgi = (X + ref_JND).clip(-meana/stda, (1-meana)/stda)
    #         imgi = imgi * stda + meana
    #         imgi = (imgi[i]*255).cpu().numpy().astype(np.uint8)
    #         imgi = imgi.transpose(1,2,0)
    #         # print('imgi.dtype',imgi.dtype) #uint8
    #         # print('imgi.max()',imgi.max()) #255
    #         # print('imgi.min()',imgi.min()) #0
    #         # print('imgi',imgi)
    #         # if not os.path.exists('/home/ycx/program/RobustIQA/attack_example/results_test'):
    #         #     print('Not exist!')
    #         cv2.imwrite('/home/ycx/program/RobustIQA/attack_example/results_test/{}_init.jpg'.format(i),imgi[:,:,::-1]) #.format(i)
            
    #         imgjndi = X_JND * stda + meana
    #         imgjndi = (imgjndi[i]*255).cpu().numpy().astype(np.uint8)
    #         imgjndi = imgjndi.transpose(1,2,0)
    #         cv2.imwrite('/home/ycx/program/RobustIQA/attack_example/results_test/{}_jnd.jpg'.format(i),imgjndi[:,:,::-1]) #.format(i)
            
    #         print('One saved.')
    #         # plt.imsave(imgi,'/home/ycx/program/RobustIQA/attack_example/results_test/{}_init.jpg'.format(i))
    
    stop_attack = stop_attack + stop_attack1 #一个是true则全true
    print('stop_attack',stop_attack)
    return init, boundary, stop_attack, already_rand_t

def expand_vector(x, size):
    batch_size = x.size(0)
    x = x.view(-1, 3, size, size)
    return x
    
def get_init_with_noise_init_attempt(model, X, y, X_ori=None, max_MOS=None, ref=None, \
                        boundary=None, last_boundary=None, X_JND=None, X_edges=None,  \
                        X_sals = None, stop_attack=None, increase=False): #, original_p=None
    max_queries = 500
    # boundary可变
    adptive_boundary = True # False #
    if adptive_boundary:
        print('Using Adaptive Boundary')
    else:
        print('Using Fixed Boundary')
    simba = True # whether use simba
    if simba:
        print('Using SimBA.')
        img_size = X.shape[-2] # 假定长宽相同
        indices = torch.randperm(3 * img_size * img_size)[:max_queries]
        n_dims = 3 * img_size * img_size
        epsilon = 5/255
        init_pert_best = torch.zeros_like(X).cuda()
    print('in get_init_with_noise')
    init = X.clone()# (a,b) torch.Size([2, 3, 224, 224])
    # print('y',y) # shape:(B)
    # stop_attack = torch.tensor([False]*X.shape[0])

    # # 观察100次里，被攻击成功的次数
    # counts = 0
    # for i in range(100):
    #     init = (X + 0.15*torch.randn_like(X)).clip(0, 1)
    #     # torch.where(
    #     #     atleast_kdim(p == y, len(X.shape)), 
    #     #     (X + 0.5*torch.randn_like(X)).clip(0, 1), 
    #     #     init)
    #     p = model(init) #.argmax(1)
    #     count = (p-y).sum()
    #     # print('count',count) #tensor(2)
    #     # print('count',count.shape) # torch.Size([])
    #     counts += torch.abs(count).detach().cpu()
    # print('counts',counts) # 0.5: 9548.7832 0.1: 493.3960 0.2: 1875.3552 0.15:1038.0006
    # os._exit(0)

    trans_in_plus_minus = True
    if trans_in_plus_minus:
        print('Using coefficients of ref_im ranging from -1 to 1')

    if increase:
        use_color_match = True #False # 
    else:
        use_color_match = False
    # use_color_match = True #For CLIVE!
    # print('For CLIVE:')

    if use_color_match:
        print('Using color match in ref_im')
    else:
        print('No color match used')

    p_ori = model(X) ###  torch.Size([8])
    if increase:
        condiction = (p_ori <= y+(abs(max_MOS-y))*boundary) #(y+max_MOS)/2 #y+(max_MOS-y)/100 /40
    else:
        condiction = (p_ori >= y-(abs(max_MOS-y))*boundary)
    condiction = condiction * (~stop_attack) #stop_attack为True的地方，cond改为False，用来不再检验
    # print('condiction',condiction) #torch.Size([8])
    stop_attack1 = torch.tensor([False]*X.shape[0]).cuda()
    p_last = p_ori

    # 更新JND_mask
    perturb_already = X - X_ori
    perturb_already_minus = torch.where(perturb_already<0,perturb_already,0)
    perturb_already_plus = torch.where(perturb_already>0,perturb_already,0)
    X_JND_minus_rest = -X_JND - perturb_already_minus
    X_JND_plus_rest = X_JND - perturb_already_plus

    meana = torch.ones_like(X).cuda() #torch.Size([2, 3, 224, 224])
    stda = torch.ones_like(X).cuda()
    meana[:,0,:,:]=0.485
    meana[:,1,:,:]=0.456
    meana[:,2,:,:]=0.406
    stda[:,0,:,:]=0.229
    stda[:,1,:,:]=0.224
    stda[:,2,:,:]=0.225
    
    if ref is not None: #(1,3,224,224)
        print('ref in get_init_with_noise.')
        if ref.shape[0]==1:
            ref = ref.repeat(X.shape[0],1,1,1) #->([3, 3, 224, 224])
            random_ref = False
            print('One ref in get_init_with_noise.')
        elif ref.shape[0]==X.shape[0]:
            random_ref = False
            print('Specific ref of each one in get_init_with_noise.')
        else:
            random_ref = True
            print('{} ref in get_init_with_noise.'.format(len(ref)))
    else:
        random_ref = False
        print('Using 0.5*Gnoise in get_init_with_noise.')
        print('Using 1 or -1 in row in get_init_with_noise.')
    if X_edges is not None:
        print('Using X_edges in get_init_with_noise.')
    if X_sals is not None:
        print('Using X_sals in get_init_with_noise.')

    random_times = 0
    while any(condiction): #p == y #若有一个True则True，否则False #torch.Size([8])
        ### 减小boundary的机制 ###
        random_times+=1
        if random_times>max_queries: # 减小难以搜到x_b的boundary
            if adptive_boundary:
                stop_attack1 = condiction * ((boundary-last_boundary)<=1/400) #对boundary<=1/200，且max_queries步没有找到的，标记停止计算
                print('stop_attack1',stop_attack1)
                if stop_attack1.any(): # 即当前已经达到了最小的boundary，但搜了200次仍然搜不到
                    print('One case cannot find x_b, boundary:',boundary)
                    condiction = condiction * (~stop_attack1)# stop_attack里是true的地方，在condition中标为fasle
                    if not any(condiction): #如果只剩这一个有问题的了
                        print('Break.')
                        break
                # 若减小后,该iter的boundary（即(boundary-last_boundary)）>1/200,则减小；否则标记停止
                inf_boundary = last_boundary + (boundary-last_boundary)/2
                desc_mask = condiction * ((boundary-last_boundary)>1/400) # 减小boundary的标准：200次没有找到x_b,且当前boundary>1/200
                boundary = boundary + desc_mask*(inf_boundary-boundary) # boundary减小
                print('One boundary decreased case, changed boundary:',boundary)
                random_times = 1
            else: #固定边界
                stop_attack1 = condiction #torch.tensor([True]*condiction.shape[0]).cuda()
                condiction = torch.tensor([True]*condiction.shape[0]).cuda()
                print('One stop for fixed boundary.')
                break
        #######
            
        # 设置攻击的init
        if max_MOS is not None:
            randi = torch.rand(X.shape[0]).unsqueeze(1).unsqueeze(1).unsqueeze(1).cuda() #->([3, 1, 1, 1])
            if trans_in_plus_minus:
                randi = (randi-0.5)*2
            if random_ref:
                ref_choice = random.choice(range(len(ref)))
                rand_ref = 0.5*randi*ref[ref_choice].repeat(X.shape[0],1,1,1)
                region_mask = None
                if X_edges is not None:
                    region_mask = torch.where(X_edges>0,1,0)
                if X_sals is not None:
                    region_mask2 = torch.where(X_sals>1,1,0)
                    if X_edges is not None:
                        region_mask = torch.where(region_mask+region_mask2>0.5,1,0)
                    else:
                        region_mask = region_mask2
                # if X_edges is not None:   
                #     rand_ref = torch.where(X_edges>0,rand_ref,0)
                if region_mask is not None:
                    rand_ref = torch.where(region_mask>0,rand_ref,0)
            else:
                if ref is not None:
                    rand_ref = 0.5*randi*ref #([3, 3, 224, 224]) #0.2 0.5
                else: # TODO 主要改这一个地方
                    # 高斯初始化
                    # rand_ref = 0.5*torch.rand_like(X) 
                    # 条状初始化
                    # B,C,H,W = X.shape
                    # rand_ref = torch.randint(0, 2, (B,C,1,W))-1
                    # rand_ref = (rand_ref*2/255).float().cuda()
                    # rand_ref = (rand_ref)/stda
                    # SimBA
                    dim = indices[random_times-1]
                    diff = torch.zeros(X.shape[0], n_dims).cuda()
                    diff[:, dim] = epsilon
                    rand_ref = expand_vector(diff,img_size)/stda + init_pert_best       
                    
                # rand_mask = torch.rand_like(X)
                # rand_mask = torch.where(rand_mask>0.5,1,-1)
                # rand_mask = rand_mask*5/255
                # rand_mask = (rand_mask)/stda # - meana
                # rand_ref = torch.where(ref>0, rand_mask,0)

            if use_color_match:
                rand_ref = color_match(X,rand_ref).cuda() #torch.Size([8, 3, 224, 224]) -> torch.Size([8, 3, 224, 224])
                
            # # 试一下使用（-1，1）的噪声
            # rand_mask = torch.rand_like(X)
            # rand_mask = torch.where(rand_mask>0.5,1,-1)
            # rand_mask = rand_mask*5/255
            # rand_mask = (rand_mask - meana)/stda
            # rand_ref = rand_mask

            rand_ref_plus = torch.where(rand_ref>0,rand_ref,0)
            rand_ref_minus = torch.where(rand_ref<0,rand_ref,0)
            ref_JND_plus = torch.where(X_JND_plus_rest>rand_ref_plus, rand_ref_plus, X_JND_plus_rest) #截断，大于 #X_JND
            ref_JND_minus = torch.where(X_JND_minus_rest<rand_ref_minus, rand_ref_minus, X_JND_minus_rest) #截断，小于JND下界的，截断为下界 #X_JND
            ref_JND = ref_JND_plus + ref_JND_minus
            # print('(X_JND-abs(ref_JND)).min()',(X_JND-abs(ref_JND)).min()) #有问题 一直都是负的
            # print('(X_JND-abs(X_JND_plus_rest)).min()',(X_JND-abs(X_JND_plus_rest)).min()) # 没问题
            # print('(X_JND-abs(X_JND_minus_rest)).min()',(X_JND-abs(X_JND_minus_rest)).min()) # 没问题
            
            X1=(X + ref_JND).clip(-meana/stda, (1-meana)/stda)
            if simba:
                rand_ref_plus = torch.where(-rand_ref>0,rand_ref,0)
                rand_ref_minus = torch.where(-rand_ref<0,rand_ref,0)
                ref_JND_plus = torch.where(X_JND_plus_rest>rand_ref_plus, rand_ref_plus, X_JND_plus_rest) #截断，大于 #X_JND
                ref_JND_minus = torch.where(X_JND_minus_rest<rand_ref_minus, rand_ref_minus, X_JND_minus_rest) #截断，小于JND下界的，截断为下界 #X_JND
                ref_JND = ref_JND_plus + ref_JND_minus
                X2 = (X - ref_JND).clip(-meana/stda, (1-meana)/stda)

            init = torch.where(
                atleast_kdim(condiction, len(X.shape)), # p == y # 用于维持X的形状 
                X1, #(X + ref_JND).clip(-meana/stda, (1-meana)/stda), #0.5 (4/255)*ref torch.randn_like(X) 0.2*randi*ref
                init)
            if simba:
               init2 = torch.where(
                atleast_kdim(condiction, len(X.shape)), # p == y # 用于维持X的形状 
                X2, #(X + ref_JND).clip(-meana/stda, (1-meana)/stda), #0.5 (4/255)*ref torch.randn_like(X) 0.2*randi*ref
                init) 
        else:
            init = torch.where(
                atleast_kdim(condiction, len(X.shape)), # p == y # 用于维持X的形状 
                (X + (4/255)*torch.randn_like(X)).clip(0, 1), #0.5
                init)
            
        p = model(init)
        if simba:
            init_pert1 = init-X
            p2 = model(init2)
            init_pert2 = init2-X
            if increase:
                cond1 = (p > p_last)
                init_pert_best = torch.where(atleast_kdim(cond1, len(X.shape)),init_pert1,init_pert_best)
                cond2 = (p2 > p_last) & (p2 > p)
                init_pert_best = torch.where(atleast_kdim(cond2, len(X.shape)),init_pert2,init_pert_best)
            else:
                cond1 = (p < p_last)
                init_pert_best = torch.where(atleast_kdim(cond1, len(X.shape)),init_pert1,init_pert_best)
                cond2 = (p2 < p_last) & (p2 < p)
                init_pert_best = torch.where(atleast_kdim(cond2, len(X.shape)),init_pert2,init_pert_best)
            init = init_pert_best + X

        ### 检验是否需要增加boundary ###
        if adptive_boundary:
            sup_boundary = torch.max(boundary + (boundary-last_boundary)*2,torch.tensor(0.01))
            if increase:
                incr_mask = p > y+(abs(max_MOS-y))*sup_boundary #增加boundary的条件：得到的分数大于两倍的boundary
            else:
                incr_mask = p < y-(abs(max_MOS-y))*sup_boundary #增加boundary的条件：得到的分数小于两倍的boundary
            # incr_mask = torch.where(boundary>=0.8,False,incr_mask) # boundary>=0.4的地方，停止增加(incr_mask=False)

            if incr_mask.any(): # 有任何一个样本满足增加条件，则更改boundary
                print('sup_boundary',sup_boundary)
                print('boundary0',boundary)
                boundary = boundary + incr_mask*(sup_boundary-boundary)/2 #boundary -> 1.5*boundary
                print('boundary1',boundary) # boundary1和boundary0应该是不一样的
                print('One boundary increased case, changed boundary:',boundary)
        #######
        if increase:
            condiction = (p <= y+(abs(max_MOS-y))*boundary) #(y+max_MOS)/2  #y+(max_MOS-y)/100 /40  
        else:
            condiction = (p >= y-(abs(max_MOS-y))*boundary)
        condiction = condiction * (~stop_attack) #torch.Size([8])
        # print('condition',condiction)
        p_last = p
        print('p',p)
        print('random_times',random_times)
    
    # print('for attack output')
    # for i,booli in enumerate(condiction): # 没问题，在范围内
    #     if not booli:
    #         imgi = init[i]-X_ori[i]
    #         diffi = X_JND[i] - abs(imgi)
    #         # print('X_JND_plus_rest[i].max()',X_JND_plus_rest[i].max()) 
    #         # print('X_JND_minus_rest[i].min()',X_JND_minus_rest[i].min())
    #         print('X_JND.max()',X_JND[i].max())
    #         print('imgi.max()',imgi.max()) #第二次就有问题了
    #         print('diffi.min()',diffi.min()) #第一次就有问题了
    #         print('imgi.min()',imgi.min())
            
    stop_attack = stop_attack + stop_attack1 #一个是true则全true
    print('stop_attack',stop_attack)
    return init, boundary, stop_attack
