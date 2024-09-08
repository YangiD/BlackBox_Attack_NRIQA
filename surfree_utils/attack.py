import torch
from surfree_utils.utils import atleast_kdim
import random

def get_init_with_noise(model, X, y, X_ori=None, max_MOS=None, ref=None, \
                        boundary=None, last_boundary=None, X_JND=None, X_edges=None,  \
                        X_sals = None, stop_attack=None, increase=False, already_rand_t=None):
    max_queries = 200
    adptive_boundary = True
    trans_in_plus_minus = True
    init = X.clone()

    p = model(X)
    if increase:
        condiction = (p <= y+(abs(max_MOS-y))*boundary)
    else:
        condiction = (p >= y-(abs(max_MOS-y))*boundary)
    condiction = condiction * (~stop_attack)
    stop_attack1 = torch.tensor([False]*X.shape[0]).cuda() 

    # update JND_mask
    perturb_already = X - X_ori
    perturb_already_minus = torch.where(perturb_already<0,perturb_already,0)
    perturb_already_plus = torch.where(perturb_already>0,perturb_already,0)
    X_JND_minus_rest = -X_JND - perturb_already_minus
    X_JND_plus_rest = X_JND - perturb_already_plus

    meana = torch.ones_like(X).cuda()
    stda = torch.ones_like(X).cuda()
    meana[:,0,:,:]=0.485
    meana[:,1,:,:]=0.456
    meana[:,2,:,:]=0.406
    stda[:,0,:,:]=0.229
    stda[:,1,:,:]=0.224
    stda[:,2,:,:]=0.225

    if ref is not None:
        if ref.shape[0]==1:
            ref = ref.repeat(X.shape[0],1,1,1)
            random_ref = False
            # print('One ref in get_init_with_noise.')
        elif ref.shape[0]==X.shape[0]:
            random_ref = False
            # print('Specific ref of each one in get_init_with_noise.')
        else:
            random_ref = True
            scale = 0.1
            # print('{} ref in get_init_with_noise.'.format(len(ref)))
    else:
        raise NotImplementedError

    # Random Search
    random_times = 0
    while any(condiction):
        random_times+=1
        if random_times>max_queries: # Adjust the boundary
            if adptive_boundary:
                stop_attack1 = condiction * ((boundary-last_boundary)<=1/400)
                if stop_attack1.any():
                    # print('One case cannot find x_b, boundary:',boundary)
                    condiction = condiction * (~stop_attack1)
                    if not any(condiction): # All samples are stopped
                        # print('Break.')
                        break
                inf_boundary = last_boundary + (boundary-last_boundary)/2
                desc_mask = condiction * ((boundary-last_boundary)>1/400)
                boundary = boundary + desc_mask*(inf_boundary-boundary) # Increase the strength of boundary
                # print('One boundary decreased case, changed boundary:',boundary)
                random_times = 1
                already_rand_t += max_queries
                if already_rand_t > 6000:
                    # print('Break.: already_rand_t reaches maximum.')
                    stop_attack1 = condiction
                    condiction = torch.tensor([True]*condiction.shape[0]).cuda()
                    break
            else:
                stop_attack1 = condiction
                condiction = torch.tensor([True]*condiction.shape[0]).cuda()
                # print('One stop for fixed boundary.')
                break
        #######
            
        randi = torch.rand(X.shape[0]).unsqueeze(1).unsqueeze(1).unsqueeze(1).cuda()
        if trans_in_plus_minus:
            randi = (randi-0.5)*2
        if random_ref:
            ref_choice = random.choice(range(len(ref)))
            rand_ref = scale * randi * ref[ref_choice].repeat(X.shape[0],1,1,1)
            region_mask = None
            if X_edges is not None:
                region_mask = torch.where(X_edges>0,1,0)
            if X_sals is not None:
                region_mask2 = torch.where(X_sals>1,1,0)
                if X_edges is not None:
                    region_mask = torch.where(region_mask+region_mask2>0.5,1,0)
                else:
                    region_mask = region_mask2
            if region_mask is not None:
                rand_ref = torch.where(region_mask>0,rand_ref,0)
        else:
            if ref is not None:
                rand_ref = scale * randi * ref
            else:
                raise NotImplementedError    

            
        rand_ref_plus = torch.where(rand_ref>0,rand_ref,0)
        rand_ref_minus = torch.where(rand_ref<0,rand_ref,0)
        ref_JND_plus = torch.where(X_JND_plus_rest>rand_ref_plus, rand_ref_plus, X_JND_plus_rest)
        ref_JND_minus = torch.where(X_JND_minus_rest<rand_ref_minus, rand_ref_minus, X_JND_minus_rest)
        ref_JND = ref_JND_plus + ref_JND_minus
        X1=(X + ref_JND).clip(-meana/stda, (1-meana)/stda)
        init = torch.where(
            atleast_kdim(condiction, len(X.shape)),
            X1,
            init)
        p = model(init)
        
        if adptive_boundary:
            sup_boundary = torch.max(boundary + (boundary-last_boundary)*2,torch.tensor(0.005))
            if increase:
                incr_mask = p > y+(abs(max_MOS-y))*sup_boundary
            else:
                incr_mask = p < y-(abs(max_MOS-y))*sup_boundary
            
            if incr_mask.any():
                boundary = boundary + incr_mask*(sup_boundary-boundary)/2
        if increase:
            condiction = (p <= y+(abs(max_MOS-y))*boundary) 
        else:
            condiction = (p >= y-(abs(max_MOS-y))*boundary)
        condiction = condiction * (~stop_attack)
        

    already_rand_t += random_times
            
    stop_attack = stop_attack + stop_attack1
    return init, boundary, stop_attack, already_rand_t

def expand_vector(x, size):
    batch_size = x.size(0)
    x = x.view(-1, 3, size, size)
    return x