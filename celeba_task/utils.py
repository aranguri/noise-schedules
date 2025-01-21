import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
def store(var, num_steps, b):
    f   = f'/scratch/sa7270/env2/celeba3/data/{var}_{num_steps}.npy'
    old = np.load(f, allow_pickle=True) if Path(f).exists() else []
    np.save(f[:-4], np.concatenate((old, b)))

def load(var, num_steps):
    f   = f'/scratch/sa7270/env2/celeba3/data/{var}_{num_steps}.npy'
    return np.load(f, allow_pickle=True) if Path(f).exists() else []

def save_low(result, var, n):
    f   = f'/scratch/sa7270/env2/celeba3/data/{var}_{n}_low.npy'
    result = np.concatenate((np.load(f, allow_pickle=True), result)) if Path(f).exists() else result
    np.save(f[:-4], result)
    
def save_high(result, var, n):
    f   = f'/scratch/sa7270/env2/celeba3/data/{var}_{n}_high.npy'
    result = np.concatenate((np.load(f, allow_pickle=True), result)) if Path(f).exists() else result
    np.save(f[:-4], result)

def load_low(var, n):
    f   = f'/scratch/sa7270/env2/celeba3/data/{var}_{n}_low.npy'
    return np.load(f, allow_pickle=True) if Path(f).exists() else []

def load_high(var, n):
    f   = f'/scratch/sa7270/env2/celeba3/data/{var}_{n}_high.npy'
    return np.load(f, allow_pickle=True) if Path(f).exists() else []

def store_imgs(var, num_steps, imgs):
    imgs = np.array([np.array(i) for i in imgs[0]])
    f   = f'/scratch/sa7270/env2/celeba3/data/{var}_{num_steps}_imgs.npy'
    imgs = np.concatenate((np.load(f, allow_pickle=True), imgs)) if Path(f).exists() else imgs
    np.save(f[:-4], imgs)

def load_imgs(var, num_steps):
    f   = f'/scratch/sa7270/env2/celeba3/data/{var}_{num_steps}_imgs.npy'
    return np.load(f, allow_pickle=True) if Path(f).exists() else []

def prepro(x):
    x = x.transpose((1,2,0))
    return x.squeeze()
    
def show(x):
    plt.imshow(x)
    plt.show()


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def setupp():
    n = 8  # Number of images
    
    # Create a figure and a GridSpec layout (one row for text and images)
    fig = plt.figure(figsize=(3*n + 2, 1))  # Adjust the figure size
    gs = GridSpec(1, n + 1, width_ratios=[3/2, *[2]*n])  # Control column sizes

    # Add "Num steps:" in the first column
    ax_num = fig.add_subplot(gs[0, 0])
    ax_num.text(0.5, 0.5, 'Num\nsteps', fontsize=37, ha='center', va='center')
    ax_num.axis('off')  # Remove axis for text display
    
    # Add "Generated images:" across all image columns
    ax_title = fig.add_subplot(gs[0, 1:])
    ax_title.text(0.5, 0.5, 'Generated images using VE SDE', fontsize=37, ha='center', va='center')
    ax_title.axis('off')  # Remove axis for text display
    
   
    
    # Adjust layout: no space between images, and the number column is narrower
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.tight_layout(pad=0)
    #plt.show()
    plt.savefig('begin.png')


def show_many(x, number):
    n = len(x)
    
    # Create a figure and a GridSpec layout (one row for number and images)
    #fig = plt.figure(figsize=(2*n + 2, 2))  # Adjust the figure size
    fig = plt.figure(figsize=(3*n + 2, 3))
    gs = GridSpec(1, n + 1, width_ratios=[3/2, *[2]*n])  # Control column sizes

    # Add number in the first column (left side)
    ax_num = fig.add_subplot(gs[0, 0])
    ax_num.text(0.5, 0.5, f'{number}', fontsize=28, ha='center', va='center')
    ax_num.axis('off')  # Remove axis for number display
    
    # Add image subplots in the remaining columns
    for i in range(n):
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(x[i], cmap='gray')
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
        ax.set_frame_on(False)  # Remove frame around the image
    
    # Adjust layout: no space between images, and the number column is narrower
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.tight_layout(pad=0)
    #plt.show()
    plt.savefig(f'{number}.png')