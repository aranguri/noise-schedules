import sys
sys.path.insert(0, '/scratch/sa7270/p4')
import torch
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline
from utils import store, load, store_imgs
model_id = 'google/ddpm-celebahq-256'
pipe = DDPMPipeline.from_pretrained(model_id) 
device = torch.device('cuda:0')
pipe = pipe.to(device)

batch_size = 40

while True:
    for n in [500]:
        imgs  = pipe(batch_size=batch_size, num_inference_steps=n)
        store_imgs('vp', n, imgs)