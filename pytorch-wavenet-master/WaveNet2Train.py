
# coding: utf-8

# # Train WaveNet

# In[1]:

from wavenet_model import WaveNetModel, ExpandingWaveNetModel
from wavenet_training import AudioFileLoader, WaveNetOptimizer

import torch
import numpy as np
import time

from IPython.display import Audio
from matplotlib import pyplot as plt
from matplotlib import pylab as pl
from IPython import display

get_ipython().magic('matplotlib notebook')


# ## Setup Model

# In[2]:

train_samples = ["train_samples/clarinet_g.wav"]
sampling_rate = 11025
init_model = None

layers = 5
blocks = 2
classes = 256
dilation_channels = 32
residual_channels = 32
skip_channels = 64
kernel_size = 2
output_length = 2
dtype = torch.FloatTensor
ltype = torch.LongTensor

use_cuda = torch.cuda.is_available()
if use_cuda:
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor


# In[3]:

model = ExpandingWaveNetModel(layers=layers,
                              blocks=blocks,
                              dilation_channels=dilation_channels,
                              residual_channels=residual_channels,
                              skip_channels=skip_channels,
                              classes=classes,
                              output_length=output_length,
                              kernel_size=kernel_size,
                              dtype=dtype)

if use_cuda:
    model.cuda()
    print("use cuda")

#print("model: ", model)
print("receptive_field: ", model.receptive_field)
print("parameter count: ", model.parameter_count())

if init_model != None:
    if use_cuda:
        model.load_state_dict(torch.load(init_model))
    else:
        # move to cpu
        model.load_state_dict(torch.load(init_model, map_location=lambda storage, loc: storage))

data_loader = AudioFileLoader(train_samples,
                              classes=classes,
                              receptive_field=model.receptive_field,
                              target_length=model.output_length,
                              dtype=dtype,
                              ltype=ltype,
                              sampling_rate=sampling_rate)


# In[4]:

from visualize import make_dot
from torch.autograd import Variable

input = Variable(torch.rand(1, 1, 256))
output = model(input)
params = dict(model.named_parameters())
#output.backward()
print(output)

make_dot(output, params)


# In[5]:


print("output length: ",  model.output_length)

data_loader.start_new_epoch()
start_data = data_loader.get_wavenet_minibatch([model.receptive_field], 
                                               model.receptive_field,
                                               model.output_length)[0]
start_data = start_data.squeeze()

plt.ion()
plt.plot(start_data[-200:].numpy())
plt.ioff()



# ## Train Model

# In[ ]:

learning_rate = 0.03
mini_batch_size = 32
report_interval = 4
validation_interval = 64
snapshot_interval = 512
epochs = 250
segments_per_chunk=16
examples_per_segment=32
validation_segments = 8
examples_per_validation_segment = 8
model_path = "model_parameters/clarinet_g_7-3-256-32-32-64-2"


# In[ ]:

def report_callback(opt):
    ax.clear()
    ax.grid(linestyle="--", axis="y")
    ax.plot(opt.loss_positions, opt.losses)
    ax.plot(opt.validation_result_positions, opt.validation_results)
    fig.canvas.draw()
    
#def test_callback(test_results, positions):
    
optimizer = WaveNetOptimizer(model,
                             data=data_loader,
                             validation_segments=validation_segments,
                             examples_per_validation_segment=examples_per_validation_segment,
                             report_callback=report_callback,
                             report_interval=report_interval,
                             validation_interval=validation_interval,
                             validation_report_callback=None,
                             snapshot_interval=snapshot_interval,
                             snapshot_file=model_path)


# In[ ]:

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()

fig.show()
fig.canvas.draw()

print('start training...')
tic = time.time()
optimizer.train(learning_rate=learning_rate,
                minibatch_size=mini_batch_size,
                epochs=epochs,
                segments_per_chunk=segments_per_chunk,
                examples_per_segment=examples_per_segment)
toc = time.time()
print('Training took {} seconds.'.format(toc-tic))


# In[ ]:

optimizer.step_times


# In[ ]:

torch.save(model.state_dict(), model_path)


# In[6]:

a = torch.rand(4)
for i, v in enumerate(a):
    print(i, v)


# In[ ]:



