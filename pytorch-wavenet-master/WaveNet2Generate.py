
# coding: utf-8

# # WaveNet Sample Generation
# Fast generation of samples from a pretrained WaveNet model

# In[37]:

from wavenet_model import WaveNetModel
from wavenet_training import AudioFileLoader, WaveNetOptimizer

import torch
import numpy as np
import time

from IPython.display import Audio
from matplotlib import pyplot as plt
from matplotlib import pylab as pl
from IPython import display

get_ipython().magic('matplotlib notebook')


# ## Load Model

# In[38]:

train_samples = ["train_samples/clarinet_g.wav"]
sampling_rate = 11025

parameters ="model_parameters/clarinet_g_7-3-256-32-32-64-2"

layers = 7
blocks = 3
classes = 256
dilation_channels = 32
residual_channels = 32
skip_channels = 64
kernel_size = 2
dtype = torch.FloatTensor
ltype = torch.LongTensor

use_cuda = torch.cuda.is_available()
if use_cuda:
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor


# In[39]:

model = WaveNetModel(layers=layers,
                     blocks=blocks,
                     dilation_channels=dilation_channels,
                     residual_channels=residual_channels,
                     skip_channels=skip_channels,
                     classes=classes,
                     kernel_size=kernel_size,
                     dtype=dtype)

if use_cuda:
    model.cuda()
    print("use cuda")

#print("model: ", model)
print("receptive_field: ", model.receptive_field)

if use_cuda:
    model.load_state_dict(torch.load(parameters))
else:
    # move to cpu
    model.load_state_dict(torch.load(parameters, map_location=lambda storage, loc: storage))

data_loader = AudioFileLoader(train_samples,
                              classes=classes,
                              receptive_field=model.receptive_field,
                              target_length=model.output_length,
                              dtype=dtype,
                              ltype=ltype,
                              sampling_rate=sampling_rate)


# In[40]:

data_loader.start_new_epoch()
data_loader.load_new_chunk()
data_loader.use_new_chunk()
start_data = data_loader.get_minibatch(1)[0]
start_data = start_data.squeeze()
#start_tensor = torch.zeros((model.scope)) + 0.0

plt.plot(start_data.cpu().numpy()[:])


# ## Generate Samples
# 

# In[41]:

num_samples = 10000 # number of samples that will be generated
out_file = "generated_samples/violin_7-2-128-32-32-32-2.wav"


# In[42]:

from ipywidgets import FloatProgress
from IPython.display import display
progress = FloatProgress(min=0, max=100)
display(progress)

def p_callback(i, total):
    progress.value += 1

tic = time.time()
generated_sample = model.generate_fast(num_samples, 
                                       first_samples=start_data,
                                       #first_samples=torch.zeros((1)),
                                       progress_callback=p_callback,
                                       sampled_generation=False,
                                       temperature=1.0)
toc = time.time()
print('Generating took {} seconds.'.format(toc-tic))


# In[43]:

fig = plt.figure()
plt.plot(generated_sample)

from IPython.display import Audio
Audio(np.array(generated_sample), rate=sampling_rate)


# In[ ]:

print(np.array(generated_sample))

from scipy.io import wavfile
wavfile.write(out_file, sampling_rate, np.array(generated_sample))


# In[ ]:



