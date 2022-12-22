'''
Author: J. Rafid Siddiqui
Azad-Academy
jrs@azaditech.com
https://www.azaditech.com

'''

import sys
from matplotlib import pyplot as plt
import numpy as np
import math
import matplotlib
from ipywidgets import *
from IPython.display import display, clear_output, Image, HTML


def sigmoid(z):
    return 1/(1 + np.exp(-z))
def sigmoid_grad(z):
    return sigmoid(z)*(1-sigmoid(z))


def weights2matrices(weights):

    W1 = weights[0:9]
    W1 = W1.reshape((3,3))
    W2 = weights[9:13]
    W2 = W2.reshape((1,4))

    return W1,W2


def weights2matrices2(weights):

    W1 = weights[0:15]
    W1 = W1.reshape((5,3))
    W2 = weights[15:45]
    W2 = W2.reshape((5,6))
    W3 = weights[45:51]
    W3 = W3.reshape((1,6))

    return W1,W2,W3

display(HTML(
    '<style>'
        '#notebook { padding-top:0px !important; } ' 
        '.container { width:100% !important; } '
        '.end_space { min-height:0px !important; } '
    '</style>'
))


def plot_data(X,Y,model=None,canvas=None,xtitle=None,ytitle=None,colors=None,plt_title=None,color_map=plt.cm.RdBu):
        
    
    if(colors is None):
        colors = np.random.rand(max(Y)+1,3)
    
        
    if(canvas is None):
        fig, ax = plt.subplots(figsize=(11,8))
    else:
        ax = canvas
        ax.cla()
    
    if(plt_title is not None):
        ax.set_title(plt_title)

    
    
    if(model is not None):  #Plotting the decision boundary
        h = .05 #mesh grid resolution
        
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
        
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=color_map, alpha=.8)
    
    if(X.shape[1]>2):
        ax.scatter3D(X[:,0],X[:,1],X[:,2],color=np.array(colors)[Y],alpha=0.6)  #plotting the 3D points
        ax.grid(False)
    else:
        ax.scatter(X[:,0],X[:,1],color=np.array(colors)[Y],alpha=0.6)  #plotting the 2D points
            
    if(xtitle is not None):
        ax.set_xlabel(xtitle,fontweight='bold',fontsize=16)
    
    if(xtitle is not None):
        ax.set_ylabel(ytitle,fontweight='bold',fontsize=16)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)