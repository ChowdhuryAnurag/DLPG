
# coding: utf-8

# In[1]:

import torch
import torch.nn.functional as F
from torch.autograd import Variable

get_ipython().magic('matplotlib inline')


# In[2]:

x = torch.Tensor(2, 8)
F.conv2d(x, torch.randn(2))


# In[4]:

x = torch.Tensor(1, 8)
x


# In[5]:

x = torch.rand(8, 1)
print('x: ', x)


# In[6]:

print('x as 8x1: ', x.view(8,1))
print('x as 4x2: ', x.view(4,2))
print('x as 2x4: ', x.view(2,4))


# In[7]:

[m, n] = x.size()
print(m)
print(n)


# In[8]:

x = torch.rand(8, 1)
x = x.resize_(9, 2)
print(x)
x = Variable(x)

y = Variable(torch.FloatTensor(2, 2))
print(y)

z = torch.cat((x, y), 0)
print(z)

z.data


# In[9]:

a = 2
a *= 3
a


# In[13]:

1 // 2


# In[32]:

t = torch.linspace(0, 63, steps=64)
t = t.view(1, 64)
print('1 x 64: ', t)

t = t.transpose(0, 1).contiguous()
t = t.view(32, 2)
t = t.transpose(0, 1)
print('2 x 32: ', t)

t = t.transpose(0, 1).contiguous()
t = t.view(16, 4)
t = t.transpose(0, 1)
print('4 x 16: ', t)

t = t.transpose(0, 1).contiguous()
t = t.view(8, 8)
t = t.transpose(0, 1).contiguous()
print('8 x 8: ', t)

t = t.transpose(0, 1).contiguous()
t = t.view(1, 64)

print('1 x 64: ', t)


# In[34]:

t = t[:, 5:]
print(t)


# In[6]:

t = torch.linspace(0, 15, steps=16)
s = t[-1:]
print(s)
s = t[-14:-1:1]
print(s)
s = t[-2:]
print(s)
s = s / 2
print(s)


# In[39]:

print(torch.max(s))


# In[ ]:



