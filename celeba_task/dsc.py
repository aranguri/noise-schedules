from deepface import DeepFace
import numpy as np

def dsc(imgs):
    races = []
    for i in range(len(imgs)):
        if i % 50 == 0: print(i)
        try:
            races.append(DeepFace.analyze(
              img_path = np.array(imgs[i]), 
              actions = ['race'],
            )[0]['race'])
        except:
            races.append(None)
    return races