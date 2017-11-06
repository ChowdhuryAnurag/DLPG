
# coding: utf-8

# In[4]:

import torch
from torch.autograd import Variable
import numpy as np

from wavenet_modules import constant_pad_1d


# In[28]:

t = Variable(torch.linspace(0, 23, steps=24).view(1, 3, 8))
print(t)


# In[48]:

def dilate(x, dilation):
    [n, c, l] = x.size()
    dilation_factor = dilation / n
    if dilation == n:
        return x

    # zero padding for reshaping
    new_l = int(np.ceil(l / dilation_factor) * dilation_factor)
    if new_l != l:
        l = new_l
        x = constant_pad_1d(x, new_l, dimension=2, pad_start=True)

    # reshape according to dilation
    x = x.permute(1, 2, 0).contiguous()
    print("first transpose: ", x)
    
    l = (l * n) // dilation
    n = dilation
    
    x = x.view(c, l, n)
    print("view change: ", x)
    
    x = x.permute(2, 0, 1)
    #x = x.transpose(1, 2).transpose(0, 2).contiguous()
    print("second transpose: ", x)

    return x

r = dilate(t, 2)
print(r)


# In[49]:

r2 = dilate(r, 4)
print(r2)


# In[ ]:



