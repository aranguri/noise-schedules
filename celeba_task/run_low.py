import sys
sys.path.insert(0, '/scratch/sa7270/p4')
from utils import load_imgs, show, save_low
from deepface import DeepFace
import numpy as np

def s(img):
    return DeepFace.extract_faces(
      img_path = np.array(img), 
      anti_spoofing = True
    )[0]['antispoof_score']

var = sys.argv[1]
n   = int(sys.argv[2])

imgs   = load_imgs(var, n)
amount = imgs.shape[0]

result = []
for i in range(amount):
    result.append(s(imgs[i]))
    if i % 100 == 0:
        save_low(result, var, n)
        result = []
