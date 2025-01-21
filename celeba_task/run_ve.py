import sys
sys.path.insert(0, '/scratch/sa7270/p4')
from diffusers import DiffusionPipeline
import torch
from utils import store, load, store_imgs

model_id = 'google/ncsnpp-celebahq-256'
pipe = DiffusionPipeline.from_pretrained(model_id)
device = torch.device('cuda:0')
pipe = pipe.to(device)

batch_size = 20

while True:
    for n in [1250, 1500, 2000]:
        imgs  = pipe(batch_size=batch_size, num_inference_steps=n)
        store_imgs('ve', n, imgs)