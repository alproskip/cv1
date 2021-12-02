from os import walk
from PIL import Image
import numpy as np


def per_channel_histogram(img, interval=1):
    x , y = img.shape[0], img.shape[1]
    
    histograms = {
        'red':np.zeros(256, np.int16), 
        'green':np.zeros(256, np.int16), 
        'blue':np.zeros(256, np.int16)
        }

    for i in range(x):
        for j in range(y):
            histograms['red'][img[i][j][0]] += 1
            histograms['green'][img[i][j][1]] += 1
            histograms['blue'][img[i][j][2]] += 1

    if interval == 1:
        return histograms
    
    bin_num = 256 // interval
    
    new_histograms = {
        'red':np.zeros(bin_num, np.int16), 
        'green':np.zeros(bin_num, np.int16), 
        'blue':np.zeros(bin_num, np.int16)
        }

    for i in range(bin_num):
        for j in range(interval):
            new_histograms['red'][i] += histograms['red'][i * interval + j]
            new_histograms['green'][i] += histograms['green'][i * interval + j]
            new_histograms['blue'][i] += histograms['blue'][i * interval + j]

    return new_histograms
    

def normalize_histogram(h):
    return h/np.sum(h)


def kl_divergence(query_hist, support_hist):
    total = 0
    for q, s in zip(query_hist, support_hist):
        total += q * np.log2(q/s)
    return total

  
if __name__=='__main__':
    support_filenames = next(walk('dataset/support_96'), (None, None, []))[2]
    query_1_filenames = next(walk('dataset/query_1'), (None, None, []))[2]
    query_2_filenames = next(walk('dataset/query_2'), (None, None, []))[2]
    query_3_filenames = next(walk('dataset/query_3'), (None, None, []))[2]

    support_images = []
    for filename in support_filenames:
        with Image.open('dataset/support_96/{}'.format(filename)) as image:
            support_images.append((filename, np.asarray(image)))
