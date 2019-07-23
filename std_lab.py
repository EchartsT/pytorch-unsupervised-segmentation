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
parser.add_argument('--nChannel', metavar='N', default=100, type=int,
                    help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=1000, type=int,
                    help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int,
                    help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float,
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int,
                    help='number of convolutional layers')
parser.add_argument('--num_superpixels', metavar='K', default=1000, type=int,
                    help='number of superpixels')
parser.add_argument('--compactness', metavar='C', default=100, type=float,
                    help='compactness of superpixels')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int,
                    help='visualization flag')
parser.add_argument('--input', metavar='FILENAME',
                    help='input image file name', required=True)
args = parser.parse_args()

# load image
im = cv2.imread(args.input)  #(321,481,3)

img_lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB)
im_val=[]
# im_val = 0.299 * img_lab[:, :, 0] + 0.587 * img_lab[:, :, 1] + 0.114 * img_lab[:, :, 2]
# im_val = np.array(im_val).flatten()

#每个点的像素值（一维）
# im_val=[]
# im_val = 0.299 * im[:, :, 0] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 2]
# im_val = np.array(im_val).flatten()

# slic
labels = segmentation.slic(im, compactness=args.compactness, n_segments=args.num_superpixels)
# boundary = segmentation.mark_boundaries(im, labels, (1, 0, 0))
# imsave('boundary_%d.png'%args.num_superpixels, boundary)

ave_position,red_average,green_average,blue_average,position= function.init_sp(im,labels)
labels = labels.reshape(im.shape[0]*im.shape[1]) #分割后每个超像素的Sk值
u_labels = np.unique(labels) #将Sk作为标签
l_inds = [] #每i行表示Si超像素中每个像素的编号
for i in range(len(u_labels)):
    l_inds.append( np.where( labels == u_labels[ i ] )[ 0 ] )

mean_list_L=[]
mean_list_A=[]
mean_list_B=[]
var_list_L=[]
var_list_A=[]
var_list_B=[]

# position = np.array(position).flatten()

for i in range(len(l_inds)):
    position_per_sp = position[i]
    labels_per_sp = []
    for temp in position_per_sp:
        m = temp[0]
        n = temp[1]
        # labels_per_sp.append(img_lab[temp])
        labels_per_sp.append(img_lab[m,n])
    # 求均值
    arr_mean_L = np.mean(labels_per_sp[0])
    arr_mean_A = np.mean(labels_per_sp[1])
    arr_mean_B = np.mean(labels_per_sp[2])
    # 求方差
    arr_var_L = np.var(labels_per_sp[0])
    arr_var_A = np.var(labels_per_sp[1])
    arr_var_B = np.var(labels_per_sp[2])
    # #求标准差
    # arr_std = np.std(labels_per_sp, ddof=1)
    mean_list_L.append(arr_mean_L)
    mean_list_A.append(arr_mean_A)
    mean_list_B.append(arr_mean_B)
    var_list_L.append(arr_var_L)
    var_list_A.append(arr_var_A)
    var_list_B.append(arr_var_B)

# fileObject=[]
# fileObject[0] = open('L_MEAN_List_%d.txt'%args.num_superpixels, 'w')
# fileObject[1] = open('A_MEAN_List_%d.txt'%args.num_superpixels, 'w')
# fileObject[2] = open('B_MEAN_List_%d.txt'%args.num_superpixels, 'w')
# fileObject[3] = open('L_VAR_List_%d.txt'%args.num_superpixels, 'w')
# fileObject[4] = open('A_VAR_List_%d.txt'%args.num_superpixels, 'w')
# fileObject[5] = open('B_VAR_List_%d.txt'%args.num_superpixels, 'w')
#
# for ip in std_list:
# 	fileObject.write(str(ip))
# 	fileObject.write('\n')
# fileObject.close()
# #
# x =  np.random.rand(len(l_inds))
plt.figure(num="mean")

plt.subplot(2,2,1)
plt.title("L")
x = range(1,len(l_inds)+1)
y = mean_list_L
plt.scatter(x, y, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）

plt.subplot(2,2,2)
plt.title("A")
x = range(1,len(l_inds)+1)
y = mean_list_A
plt.scatter(x, y, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）

plt.subplot(2,2,3)
plt.title("B")
x = range(1,len(l_inds)+1)
y = mean_list_B
plt.scatter(x, y, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）


plt.savefig('EVAL/LAB/mean_%d.png'%args.num_superpixels)



plt.figure(num="var")

plt.subplot(2,2,1)
plt.title("L")
x = range(1,len(l_inds)+1)
y = var_list_L
plt.scatter(x, y, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）

plt.subplot(2,2,2)
plt.title("A")
x = range(1,len(l_inds)+1)
y = var_list_A
plt.scatter(x, y, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）

plt.subplot(2,2,3)
plt.title("B")
x = range(1,len(l_inds)+1)
y = var_list_B
plt.scatter(x, y, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）


plt.savefig('EVAL/LAB/var_%d.png'%args.num_superpixels)
# plt.show()



