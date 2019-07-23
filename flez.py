#from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import numpy as np
from skimage import segmentation
import torch.nn.init
import function
from matplotlib.pyplot import imsave
import matplotlib.pyplot as plt
import os

use_cuda = torch.cuda.is_available()
parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--num_superpixels', metavar='K', default=10000, type=int,
                    help='number of superpixels')
parser.add_argument('--compactness', metavar='C', default=100, type=float,
                    help='compactness of superpixels')
parser.add_argument('--input', metavar='FILENAME',
                    help='input image file name', required=True)
parser.add_argument('--scale', metavar='S', default=10, type=int,
                    help='number of superpixels')
parser.add_argument('--min_size', metavar='MS', default=500, type=int,
                    help='min number of pixels')
#S更高意味着更大的集群
parser.add_argument('--re_scale', metavar='S', default=20, type=int,
                    help='re area')
# 最小的组件的大小。强制使用后处理
parser.add_argument('--re_min_size', metavar='MS', default=100, type=int,
                    help='re min number of pixels')

args = parser.parse_args()

# load image
im = cv2.imread(args.input)  #(321,481,3)
filename =   os.path.splitext(os.path.split(args.input)[1])[0]

im_val=[]
im_val = 0.299 * im[:, :, 0] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 2]
im_val_temp = im_val
im_val = np.array(im_val).flatten()

#每个点的像素值（一维）
im_val=[]
im_val = 0.299 * im[:, :, 0] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 2]
im_val = np.array(im_val).flatten()

# flez
labels = segmentation.felzenszwalb(im,scale=args.scale, sigma=0.8, min_size = args.min_size)
boundary = segmentation.mark_boundaries(im, labels, (1, 0, 0))
# imsave('boundary_%d.png'%args.num_superpixels, boundary)
labels_temp = labels
# ave_position,red_average,green_average,blue_average,position = function.init_sp(im,labels)
labels = labels.reshape(im.shape[0]*im.shape[1]) #分割后每个超像素的Sk值
u_labels = np.unique(labels) #将Sk作为标签
l_inds = [] #每i行表示Si超像素中每个像素的编号
for i in range(len(u_labels)):
    l_inds.append( np.where( labels == u_labels[ i ] )[ 0 ] )

mean_list=[]
var_list=[]
std_list=[]
std_m_v=[]
for i in range(len(l_inds)):
    labels_per_sp = im_val[l_inds[i]]
    # 求均值
    arr_mean = np.mean(labels_per_sp)
    # 求方差
    arr_var = np.var(labels_per_sp)
    #求标准差
    arr_std = np.std(labels_per_sp, ddof=1)

    #
    arr_s = arr_mean/(arr_mean**2)

    mean_list.append(arr_mean)
    var_list.append(arr_var)
    std_list.append(arr_std)
    std_m_v.append(arr_s)

plt.figure(num="rgb")

plt.subplot(2,2,1)
plt.title("mean")
x = range(1,len(l_inds)+1)
y = mean_list
plt.scatter(x, y, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）

plt.subplot(2,2,2)
plt.title("var")
x = range(1,len(l_inds)+1)
y = var_list
plt.scatter(x, y, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）

plt.subplot(2,2,3)
plt.title("v_s")
x = range(1,len(l_inds)+1)
y = std_m_v
plt.scatter(x, y, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）

plt.subplot(2,2,4)
plt.title("std")
plt.imshow(boundary)

# filenum = os.path.splitext(args.input)[0]
plt.savefig('EVAL/RGB/flez_%s_%d_%d.png'%(filename,args.scale, args.min_size))

liqun=[]
liqun_position=[]
for lq in range(len(std_m_v)):
    if (var_list[lq]>500):
        liqun.append(std_m_v[lq])
        liqun_position.append((lq))#离群的位置

badimage = np.array(np.ones([321,481]))*225
badimage_rgb = np.array(np.zeros([321,481,3]))
for k in liqun_position:
    pos_list = l_inds[k]
    for pos in pos_list:
        m = int(pos/481)
        n = int(pos%481)
        badimage[m,n] = im_val_temp[m,n]
        badimage_rgb[m, n] = im[m, n]

plt.figure(num="坏的")
plt.title("std")
plt.imshow(badimage)
plt.savefig('EVAL/needed/flez_resegpart_%s_%d_%d.png'%(filename,args.scale, args.min_size))

# badimage = cv2.cvtColor(badimage,cv2.COLOR_GRAY2RGB)
# labels = segmentation.felzenszwalb(im,scale=args.scale, sigma=0.5, min_size = args.min_size)
# boundary = segmentation.mark_boundaries(im, labels, (1, 0, 0))

labels_bad = segmentation.felzenszwalb(badimage_rgb,scale=args.re_scale, sigma=0.8, min_size = args.re_min_size)
labels_bad = labels_bad+labels_temp
boundarys_bad = segmentation.mark_boundaries(im, labels_bad, (1, 0, 0))

plt.figure(num="reseg")
plt.title("reseg")
plt.imshow(boundarys_bad )
plt.savefig('EVAL/reseg/reseg_%s_%d_%d_fine_%d_%d.png'%(filename,args.scale, args.min_size,args.re_scale, args.re_min_size))

# fileObject = open('std_List_%d.txt'%args.num_superpixels, 'w')
# for ip in std_list:
# 	fileObject.write(str(ip))
# 	fileObject.write('\n')
# fileObject.close()
#
# # x =  np.random.rand(len(l_inds))
# x = range(1,len(l_inds)+1)
# # y = var_list
# # plt.scatter(x, y, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
# # plt.savefig('统计_%d.png'%args.num_superpixels)
# # fig = plt.figure(0)
# y = mean_list
# plt.scatter(x, y, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
# plt.savefig('统计_mean_%d.png'%args.num_superpixels)
#
# # plt.show()



