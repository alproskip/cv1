from os import walk
from PIL import Image
import numpy as np

EPSILON = 10**(-20)

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

def color_histogram(img, interval=1):
    pch = per_channel_histogram(img, interval)
    red, green, blue = pch['red'], pch['green'], pch['blue']
    hist_size = red.shape[0]
    histogram = np.zeros((hist_size,hist_size,hist_size))
    for i in range(hist_size):
        for j in range(hist_size):
            for k in range(hist_size):
                histogram[i,j,k] = red[i]+green[j]+blue[k]
    return histogram
    
def normalize_histogram(h):
    norm = np.sum(h)
    normalized_hist = np.divide(h,np.sum(h))
    normalized_hist[normalized_hist==0] = EPSILON / norm
    return normalized_hist

def get_histogram_by_grids(img, interval=1, grid_count=1):
    grid_size = img.shape[0]//grid_count
    histogram_list = np.zeros(0)
    for i in range(grid_count):
        for j in range(grid_count):
            img_slice = img[i*grid_size:(i+1)*grid_size,j*grid_size:(j+1)*grid_size]
            slice_hist = per_channel_histogram(img_slice, interval)
            normalized_hist = {'red':normalize_histogram(slice_hist['red']), 'green':normalize_histogram(slice_hist['green']), 'blue':normalize_histogram(slice_hist['blue'])}
            histogram_list = np.append(histogram_list, normalized_hist)
    return histogram_list

def get_color_histogram_by_grids(img, interval=1, grid_count=1):
    grid_size = img.shape[0]//grid_count
    histogram_list = np.zeros(0)
    for i in range(grid_count):
        for j in range(grid_count):
            img_slice = img[i*grid_size:(i+1)*grid_size,j*grid_size:(j+1)*grid_size]
            slice_hist = color_histogram(img_slice, interval)
            normalized_hist = normalize_histogram(slice_hist)
            histogram_list = np.append(histogram_list, normalized_hist)
    return histogram_list

def kl_divergence(query_hist, support_hist):
    division = np.divide(query_hist, support_hist)
    log_div = np.log2(division)
    h_mult = np.multiply(query_hist, log_div)
    divergence = np.sum(h_mult)
    return divergence

def kl_divergence_by_grids(query_hist_list, support_hist_list):
    divergence_array = np.zeros(0)
    hist_count = query_hist_list.shape[0]
    for qh,sh in zip(query_hist_list, support_hist_list):
        divergence = kl_divergence(qh['red'], sh['red']) + kl_divergence(qh['green'], sh['green']) + kl_divergence(qh['blue'], sh['blue'])
        divergence_array = np.append(divergence_array, divergence)
    return np.average(divergence_array)

if __name__ == '__main__':
    # Read Images
    support_filenames = next(walk('dataset/support_96'), (None, None, []))[2]
    query_1_filenames = next(walk('dataset/query_1'), (None, None, []))[2]
    query_2_filenames = next(walk('dataset/query_2'), (None, None, []))[2]
    query_3_filenames = next(walk('dataset/query_3'), (None, None, []))[2]

    support_images = []
    for filename in support_filenames:
        with Image.open('dataset/support_96/{}'.format(filename)) as image:
            support_images.append((filename, np.asarray(image)))

    query_1_images = []
    for filename in query_1_filenames:
        with Image.open('dataset/query_1/{}'.format(filename)) as image:
            query_1_images.append((filename, np.asarray(image)))
    query_2_images = []
    for filename in query_2_filenames:
        with Image.open('dataset/query_2/{}'.format(filename)) as image:
            query_2_images.append((filename, np.asarray(image)))

    query_3_images = []
    for filename in query_3_filenames:
        with Image.open('dataset/query_3/{}'.format(filename)) as image:
            query_3_images.append((filename, np.asarray(image)))
            