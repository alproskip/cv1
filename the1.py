from os import walk
from PIL import Image
import numpy as np
from tqdm import tqdm

EPSILON = 10**(-20)

def per_channel_histogram(img, interval=1):
    red = np.bincount(np.ravel(img[:,:,0]), minlength=256)
    green = np.bincount(np.ravel(img[:,:,1]), minlength=256)
    blue = np.bincount(np.ravel(img[:,:,2]), minlength=256)
    histograms = {
        'red':red, 
        'green':green, 
        'blue':blue
        }
    if interval == 1:
        return histograms

    bin_num = 256//interval
    new_red = np.zeros(bin_num)
    new_green = np.zeros(bin_num)
    new_blue = np.zeros(bin_num)

    for i in range(bin_num):
        new_red[i] = np.sum(red[i*interval:(i+1)*interval])
        new_green[i] = np.sum(green[i*interval:(i+1)*interval])
        new_blue[i] = np.sum(blue[i*interval:(i+1)*interval])
        
    new_histograms = {
        'red':new_red, 
        'green':new_green, 
        'blue':new_blue
        }
        
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
    if np.sum(h) == 0:
        h[h==0] = EPSILON
        return h
    norm = np.sum(h)
    normalized_hist = np.divide(h,np.sum(h))
    normalized_hist[normalized_hist==0] = EPSILON / norm
    return normalized_hist

def get_per_channel_histogram_by_grids(img, interval=1, grid_count=96):
    grid_size = img.shape[0]//grid_count
    histogram_list = np.zeros(0)
    for i in range(grid_size):
        for j in range(grid_size):
            img_slice = img[i*grid_count:(i+1)*grid_count,j*grid_count:(j+1)*grid_count]
            slice_hist = per_channel_histogram(img_slice, interval)
            normalized_hist = {'red':normalize_histogram(slice_hist['red']), 'green':normalize_histogram(slice_hist['green']), 'blue':normalize_histogram(slice_hist['blue'])}
            histogram_list = np.append(histogram_list, normalized_hist)
    return histogram_list

def get_color_histogram_by_grids(img, interval=1, grid_count=96):
    grid_size = img.shape[0]//grid_count
    histogram_list = []
    for i in range(grid_size):
        for j in range(grid_size):
            img_slice = img[i*grid_count:(i+1)*grid_count,j*grid_count:(j+1)*grid_count]
            slice_hist = color_histogram(img_slice, interval)
            normalized_hist = normalize_histogram(slice_hist)
            histogram_list.append(normalized_hist)
    return np.asarray(histogram_list)

def kl_divergence(query_hist, support_hist):
    division = np.divide(query_hist, support_hist)
    log_div = np.log2(division)
    h_mult = np.multiply(query_hist, log_div)
    divergence = np.sum(h_mult)
    return divergence

def kl_divergence_by_grids(query_hist_list, support_hist_list):
    divergence_array = np.zeros(0)
    for qh,sh in zip(query_hist_list, support_hist_list):
        divergence = (kl_divergence(qh['red'], sh['red']) + kl_divergence(qh['green'], sh['green']) + kl_divergence(qh['blue'], sh['blue'])) / 3
        divergence_array = np.append(divergence_array, divergence)
    return np.average(divergence_array)

def kl_divergence_by_grids_color_histogram(query_hist_list, support_hist_list):
    divergence_array = np.zeros(0)
    for qh,sh in zip(query_hist_list, support_hist_list):
        divergence = kl_divergence(qh, sh)
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

    while(True):
        try:
            q_num = input("Type query number for comparison: ")
            query_images = [query_1_images, query_2_images, query_3_images][int(q_num)-1]    
            a = input("Choose a histogram type. Type p for per channel and c for color histogram:\nHistogram Type: ")
            b = input("Type quantization interval (default 1): ")
            interval = int(b) if b else 1
            c = input("Type grid dimension, press enter for default 96x96, 1 grid: ")
            grid_count = int(c) if c else 96

            if a == 'p':
                if c:
                    support_query_hist_lists = {'support_histograms':[], 'query_histograms':[]}

                    print("Reading Files...")
                    for name, img in tqdm(support_images, leave=False):
                        hist_list = get_per_channel_histogram_by_grids(img, interval=interval, grid_count=grid_count)
                        support_query_hist_lists['support_histograms'].append((name, hist_list))
                    
                    for name, img in tqdm(query_images, leave=False):
                        hist_list = get_per_channel_histogram_by_grids(img, interval=interval, grid_count=grid_count)
                        support_query_hist_lists['query_histograms'].append((name, hist_list))
                    print("Reading Files Done")
                    
                    correct_guesses = 0
                    for name, support_hist in tqdm(support_query_hist_lists['support_histograms'], leave=False):
                        min_divergence = 999999
                        for q_name, query_hist in support_query_hist_lists['query_histograms']:
                            divergence = kl_divergence_by_grids(query_hist, support_hist)
                            if divergence < min_divergence:
                                min_divergence = divergence
                                result = {'support': name, 'query': q_name, 'divergence': divergence}
                        if result['support'] == result['query']:
                            correct_guesses += 1
                    print(f"{grid_count}x{grid_count} Grid - Accuracy: {correct_guesses / 200} ")           
                else:
                    support_histograms = []
                    query_histograms = []

                    print("Reading Files...")
                    for name, img in tqdm(support_images, leave=False):
                        hist = per_channel_histogram(img, interval)
                        support_histograms.append((name, (normalize_histogram(hist['red']),normalize_histogram(hist['green']), normalize_histogram(hist['blue']))))
                    
                    for name, img in tqdm(query_images, leave=False):
                        hist = per_channel_histogram(img, interval)
                        query_histograms.append((name, (normalize_histogram(hist['red']),normalize_histogram(hist['green']), normalize_histogram(hist['blue']))))
                    print("Reading Files Done")
        
                    correct_guesses = 0
                    for name, hist in tqdm(support_histograms, leave=False):
                        min_divergence = 9999
                        for q_name, q_hist in query_histograms:
                            red_divergence = kl_divergence(q_hist[0], hist[0])
                            green_divergence = kl_divergence(q_hist[1], hist[1])
                            blue_divergence = kl_divergence(q_hist[2], hist[2])
                            divergence = (red_divergence + blue_divergence + blue_divergence) / 3
                            if divergence < min_divergence:
                                min_divergence = divergence
                                result = {'support': name, 'query': q_name, 'divergence': divergence}
                        if result['support'] == result['query']:
                            correct_guesses += 1
                    print(f"Accuracy is {correct_guesses/200} for interval: {interval}")
            
            elif a == 'c':
                if c:
                    support_query_hist_lists = {'support_histograms':[], 'query_histograms':[]}
                    print("Reading Files...")
                    for name, img in tqdm(support_images, leave=False):
                        hist_list = get_color_histogram_by_grids(img, interval=interval, grid_count=grid_count)
                        support_query_hist_lists['support_histograms'].append((name, hist_list))
                    
                    for name, img in tqdm(query_images, leave=False):
                        hist_list = get_color_histogram_by_grids(img, interval=interval, grid_count=grid_count)
                        support_query_hist_lists['query_histograms'].append((name, hist_list))
                    print("Reading Files Done")

                    correct_guesses = 0
                    support_hist_list = support_query_hist_lists['support_histograms']
                    query_hist_list = support_query_hist_lists['query_histograms']
                    for name, support_hist in tqdm(support_hist_list, leave=False):
                        min_divergence = 999999
                        for q_name, query_hist in query_hist_list:
                            divergence = kl_divergence_by_grids_color_histogram(query_hist, support_hist)
                            if divergence < min_divergence:
                                min_divergence = divergence
                                result = {'support': name, 'query': q_name, 'divergence': divergence}
                        if result['support'] == result['query']:
                            correct_guesses += 1
                    print(f"{grid_count}x{grid_count} Grid - Accuracy: {correct_guesses / 200} ")           
                
                else:
                    support_histograms = []
                    query_histograms = []

                    print("Reading Files...")
                    for name, img in tqdm(support_images, leave=False):
                        hist = normalize_histogram(color_histogram(img, interval))
                        support_histograms.append((name, hist))

                    for name, img in tqdm(query_images, leave=False):
                        hist = normalize_histogram(color_histogram(img, interval))
                        query_histograms.append((name, hist))
                    print("Reading Files Done")

                    correct_guesses = 0
                    for name, hist in tqdm(support_histograms, leave=False):
                        min_divergence = 99999
                        for q_name, q_hist in query_histograms:
                            divergence = kl_divergence(q_hist, hist)
                            if divergence < min_divergence:
                                min_divergence = divergence
                                result = {'support': name, 'query': q_name, 'divergence': divergence}
                        if result['support'] == result['query']:
                            correct_guesses += 1
                    print(f"Accuracy is {correct_guesses/200} for interval: {interval}")
            else: raise
        
        except Exception as e:
            print(e)
            print("Invalid argument given\nAvailable query numbers 1, 2 or 3\nAvailable histogram types 'p' or 'c'\n")