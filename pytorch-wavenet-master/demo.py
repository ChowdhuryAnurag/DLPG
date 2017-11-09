
# coding: utf-8

# In[9]:
## Reset workspace (optional)
print('Resetting python workspace ...')
from IPython import get_ipython
get_ipython().magic('reset -sf')

## Set the current working directory
import sys, os
import numpy as np
os.chdir('/research/iprobe-tmp/chowdh51/workspace/SRDAS/Python/DLPG/pytorch-wavenet-master')
print('Setting current working directory to ...')
print(os.getcwd())

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import time

from model import WaveNetModel, Optimizer, WaveNetData

from IPython.display import Audio
from IPython.core.debugger import Tracer
from matplotlib import pyplot as plt
from matplotlib import pylab as pl
from IPython import display
import torch
import numpy as np
#%matplotlib inline
# get_ipython().magic('matplotlib notebook')


# In[10]:

model = WaveNetModel(num_blocks=2,
              num_layers=12,
              hidden_channels=128,
              num_classes=256)
print('model: ', model)
print('scope: ', model.scope)
model = model.cuda()


# In[11]:

from scipy.io import wavfile


data = WaveNetData('../data/bach.wav',
                   input_length=model.scope,
                   target_length=model.last_block_scope,
                   num_classes=model.num_classes)
start_tensor = data.get_minibatch([30000])[0].squeeze()
plt.ion()
plt.plot(start_tensor[-200:].numpy())
plt.ioff()


# In[12]:

optimizer = Optimizer(model,
                      mini_batch_size = 1,
                      learning_rate=0.01,
                      stop_threshold=0.1,
                      avg_length=4)


# In[ ]:

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()

fig.show()
fig.canvas.draw()

def hook(losses):
    ax.clear()
    ax.plot(losses)
    fig.canvas.draw()
    #print(losses[-1])

optimizer.hook = hook

#Tracer()()

print('start training...')
tic = time.time()
optimizer.train(data)
toc = time.time()
print('Training took {} seconds.'.format(toc-tic))


# In[44]:

#torch.save(model, 'trained_model')

#start_tensor = torch.zeros((model.scope)) + 0
start_tensor = data.get_minibatch([12345])[0].squeeze()
print('generate...')
tic = time.time()
model = model.cpu()
generated = model.generate(start_data=start_tensor, num_generate=200)
# generated = model.fast_generate(40000)
toc = time.time()
print('Generating took {} seconds.'.format(toc-tic))


# In[ ]:
len(generated)
fig = plt.figure()
plt.plot(generated)
import scipy.io as sio
sio.wavfile.write('bach_regenrated_16k.wav', 16000, np.array(generated))

# In[ ]:

print('generate...')
tic = time.time()
generated = model.fast_generate(40000)
toc = time.time()
print('Generating took {} seconds.'.format(toc-tic))

fig = plt.figure()
plt.plot(generated)

len(generated)
fig = plt.figure()
plt.plot(generated)
import scipy.io as sio
sio.wavfile.write('bach_regenrated_16k.wav', 16000, np.array(generated))


# In[6]:

torch.save(model.state_dict(), "trained_state_bach_11025")


#
