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
parser.add_argument('--min_size', metavar='MS', default=100, type=int,
                    help='min number of pixels')

args = parser.parse_args()

# load image
im = cv2.imread(args.input)  #(321,481,3)

im_val=[]
im_val = 0.299 * im[:, :, 0] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 2]
im_val = np.array(im_val).flatten()

#每个点的像素值（一维）
im_val=[]
im_val = 0.299 * im[:, :, 0] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 2]
im_val = np.array(im_val).flatten()

# slic
# labels = segmentation.felzenszwalb(im,scale=args.scale, sigma=0.5, min_size = args.min_size)
labels = segmentation.slic(im, compactness=args.compactness, n_segments=args.num_superpixels)
boundary = segmentation.mark_boundaries(im, labels, (1, 0, 0))
# imsave('boundary_%d.png'%args.num_superpixels, boundary)

# ave_position,red_average,green_average,blue_average = function.init_sp(im,labels)
labels = labels.reshape(im.shape[0]*im.shape[1]) #分割后每个超像素的Sk值
u_labels = np.unique(labels) #将Sk作为标签
l_inds = [] #每i行表示Si超像素中每个像素的编号
for i in range(len(u_labels)):
    l_inds.append( np.where( labels == u_labels[ i ] )[ 0 ] )

# mean_list=[]
# var_list=[]
# std_list=[]
# for i in range(len(l_inds)):
#     labels_per_sp = im_val[l_inds[i]]
#     # 求均值
#     arr_mean = np.mean(labels_per_sp)
#     # 求方差
#     arr_var = np.var(labels_per_sp)
#     #求标准差
#     arr_std = np.std(labels_per_sp, ddof=1)
#     mean_list.append(arr_mean)
#     var_list.append(arr_var)
#     std_list.append(arr_std)


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
    #方差除以均值的平方
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

plt.savefig('EVAL/RGB/slic_12003_%d.png'%(args.num_superpixels))
# plt.figure(num="rgb")
#
# plt.subplot(2,2,1)
# plt.title("mean")
# x = range(1,len(l_inds)+1)
# y = mean_list
# plt.scatter(x, y, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
#
# plt.subplot(2,2,2)
# plt.title("var")
# x = range(1,len(l_inds)+1)
# y = var_list
# plt.scatter(x, y, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
#
# plt.subplot(2,2,3)
# plt.title("std")
# x = range(1,len(l_inds)+1)
# y = var_list
# plt.scatter(x, y, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
#
# plt.subplot(2,2,4)
# plt.title("std")
# plt.imshow(boundary)
#
# os.path.splitext(base)
# plt.savefig('EVAL/RGB/flez_%d_%d.png'%(args.scale, args.min_size))

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



