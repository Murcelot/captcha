import pandas as pd
import scipy as sp
import numpy as np

from skimage import io, color, feature, draw, morphology
from skimage.util import img_as_ubyte

import os
from tqdm import tqdm

#inputs
print('Apply Canny algorithm with smth else for images in [data_path] and saving them to [processed_data_path] in current directory')
print('Print [data_path] [data_csv_name] [processed_data_path]')
data_path, data_csv_name, processed_data_path = input().split()
base = os.path.join('.', data_path)

#data
data = pd.read_csv(os.path.join(base, data_csv_name), dtype={'solution' : str})
train_images_path_sol = data[data['image_path'].str.startswith('train')]
raw_images = []
processed_images = []
raw_images_path = os.path.join('.', processed_data_path, 'raw_images')
processed_images_path = os.path.join('.', processed_data_path, 'processed_images')
os.mkdir(raw_images_path)
os.mkdir(processed_images_path)

#segmentation functions
rect = draw.rectangle_perimeter((3,3),(47,197))
def seg(img, dilation = False):
    res = color.rgb2gray(img)
    #smoothing
    res = sp.signal.correlate2d(res, np.array([1/9] * 9).reshape(3,3))[1:51, 1:201]
    res = sp.signal.correlate2d(res, np.array([1/25] * 25).reshape(5,5))[1:51, 1:201]
    #unsupervised segmentation
    res = feature.canny(res)
    res[rect[0],rect[1]] = 0
    if dilation:
        res = morphology.binary_dilation(res)
    return sp.ndimage.binary_fill_holes(res)

def labels_check(masked_img):
    _, num_labels = sp.ndimage.label(masked_img)
    return num_labels == 6

#apply unsupervised segmentation
print('Processing...')
for i, img_path in enumerate(tqdm(train_images_path_sol['image_path'])):
    img = io.imread(os.path.join(base, img_path))
    masked_img = seg(img, True)
    if labels_check(masked_img):
        img_num = img_path[25:]
        raw_images.append(os.path.join('raw_images', img_num))
        processed_images.append(os.path.join('processed_images', img_num))
        io.imsave(os.path.join(raw_images_path, img_num), img)
        io.imsave(os.path.join(processed_images_path, img_num), img_as_ubyte(masked_img), check_contrast=False)

#saving results
print('Saving...')
df = pd.DataFrame(data = {'raw' : raw_images, 'processed' : processed_images})
df.to_csv(os.path.join('.', processed_data_path, 'segmentation_data.csv'), index=False)

print('Done!')