import torch
# from utils.utils import atleast_kdim
from surfree_utils.utils import atleast_kdim
import os


def get_init_with_noise(model, X, y):
    init = X.clone()
    p = model(X).argmax(1)

    while any(p == y):
        init = torch.where(
            atleast_kdim(p == y, len(X.shape)), 
            (X + 0.5*torch.randn_like(X)).clip(0, 1), 
            init)
        p = model(init).argmax(1)

    # # 观察100次里，被攻击成功的次数
    # counts = 0
    # for i in range(100):
    #     # init = torch.where(
    #     #     atleast_kdim(p == y, len(X.shape)), 
    #     #     (X + 0.5*torch.randn_like(X)).clip(0, 1), 
    #     #     init)
    #     init = (X + 0.15*torch.randn_like(X)).clip(0, 1)
    #     p = model(init).argmax(1)
    #     count = (p!=y).sum()
    #     # print('count',count) #tensor(2)
    #     # print('count',count.shape()) # torch.Size([])
    #     counts += count
    # print('counts',counts) # resnet18: 0.5: 500 0.1: 346 0.2: 463 resnet50: 0.2:500 0.1:258 0.15:437
    # os._exit(0)
    

    return init
