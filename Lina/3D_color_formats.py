# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import data, io, filters, color, segmentation, exposure, measure, morphology
from skimage.segmentation import felzenszwalb, slic, quickshift, random_walker
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, linkage

img = io.imread('/Users/lina/Downloads/2015-04-029_20X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_001.tif')
im1 = img[1000:1500, 1000:1500]

w, h, d = im1.shape
cell_area = 500
n_segments = int(w*h/cell_area)
labels1 = slic(im1, n_segments=n_segments, compactness=10, sigma=10, 
               multichannel=True, convert2lab=True)
out1 = color.label2rgb(labels1, im1, kind='avg')

out1_HSV = cv2.cvtColor(out1, cv2.COLOR_RGB2HSV)
out1_HLS = cv2.cvtColor(out1, cv2.COLOR_RGB2HLS)
out1_Lab = cv2.cvtColor(out1, cv2.COLOR_RGB2Lab)
out1_Luv = cv2.cvtColor(out1, cv2.COLOR_RGB2Luv)
out1_YCrCb = cv2.cvtColor(out1, cv2.COLOR_RGB2YCrCb)

colors_name = ['out1_HSV', 'out1_HLS', 'out1_Lab', 'out1_Luv', 'out1_YCrCb']
colors = [out1_HSV, out1_HLS, out1_Lab, out1_Luv, out1_YCrCb]

out1_colors = {}
i=0
for color in colors:
    X = color.reshape((-1,3))
    df = pd.DataFrame(X)
    out1_colors['%s_uniq' % colors_name[i]] = df.drop_duplicates().values #get unique color value
    i=i+1
  
#3-D plot

fig = plt.figure()
fig.suptitle('HSV', fontsize=20)
ax = fig.add_subplot(111, projection='3d')
Axes3D.scatter(ax, xs=out1_colors["out1_HSV_uniq"][:,0], ys=out1_colors["out1_HSV_uniq"][:,1], zs=out1_colors["out1_HSV_uniq"][:,2], zdir='z', s=20, c='yellow', depthshade=True)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.view_init(30,220)
plt.show()

fig = plt.figure()
fig.suptitle('HLS', fontsize=20)
ax = fig.add_subplot(111, projection='3d')
Axes3D.scatter(ax, xs=out1_colors["out1_HLS_uniq"][:,0], ys=out1_colors["out1_HLS_uniq"][:,1], zs=out1_colors["out1_HLS_uniq"][:,2], zdir='z', s=20, c='blue', depthshade=True)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.view_init(30,220)
plt.show()

fig = plt.figure()
fig.suptitle('Lab', fontsize=20)
ax = fig.add_subplot(111, projection='3d')
Axes3D.scatter(ax, xs=out1_colors["out1_Lab_uniq"][:,0], ys=out1_colors["out1_Lab_uniq"][:,1], zs=out1_colors["out1_Lab_uniq"][:,2], zdir='z', s=20, c='red', depthshade=True)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.view_init(30,220)
plt.show()

fig = plt.figure()
fig.suptitle('Luv', fontsize=20)
ax = fig.add_subplot(111, projection='3d')
Axes3D.scatter(ax, xs=out1_colors["out1_Luv_uniq"][:,0], ys=out1_colors["out1_Luv_uniq"][:,1], zs=out1_colors["out1_Luv_uniq"][:,2], zdir='z', s=20, c='green', depthshade=True)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.view_init(30,220)
plt.show()

fig = plt.figure()
fig.suptitle('YCrCb', fontsize=20)
ax = fig.add_subplot(111, projection='3d')
Axes3D.scatter(ax, xs=out1_colors["out1_YCrCb_uniq"][:,0], ys=out1_colors["out1_YCrCb_uniq"][:,1], zs=out1_colors["out1_YCrCb_uniq"][:,2], zdir='z', s=20, c='pink', depthshade=True)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.view_init(30,220)
plt.show()
