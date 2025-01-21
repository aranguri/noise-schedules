import sys
sys.path.insert(0, '/scratch/sa7270/p4')
from utils import load_imgs, save_high
from dsc import dsc
var = sys.argv[1]
n   = int(sys.argv[2])

imgs   = load_imgs(var, n)
amount = imgs.shape[0]

result = []
for i in range(amount):
    result.append(dsc([imgs[i]])[0])
    if i % 100 == 0:
        save_high(result, var, n)
        result = []