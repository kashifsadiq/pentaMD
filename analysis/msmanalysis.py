# MSManalysis functions

#Correspondence: S. Kashif Sadiq (kashif.sadiq@embl.de) Affiliation: 1. Heidelberg Institute for Theoretical Studies, HITS gGmbH 2. European Moelcular Biology Laboratory
        
#This file contains a set of python functions for Markov state model (MSM) analysis for the manuscript:

#Balogh, Gabor; Gyöngyösi, Tamás; Timári, István; Herczeg, Mihály; Borbás, Anikó; Sadiq, S. Kashif; Fehér, Krisztina; Kövér, Katalin, 
#Conformational Analysis of Heparin-Analogue Pentasaccharides by Nuclear Magnetic Resonance Spectroscopy and Molecular Dynamics Simulations (2021), 
#Journal of Chemical Information and Modeling, Accepted.


#Last modified: 13/5/2021

########################################################################################################################################

# Module dependencies


from __future__ import print_function
import pyemma

import os
#%pylab inline

import pyemma.coordinates as coor
import pyemma.msm as msm
import pyemma.plots as mplt

from pyemma import config
config.show_progress_bars = False
#print(config.show_progress_bars)

import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import gridspec
from collections import OrderedDict


import math
import numpy as np
import sys
import os.path
import random
import errno 
from shutil import copyfile
import operator
import re
from glob import glob
#from kmodes.kmodes import KModes
import random

import MDAnalysis as mda
from MDAnalysis.analysis import dihedrals
from MDAnalysis.analysis import align, rms, distances, contacts
from MDAnalysis.analysis.base import AnalysisFromFunction 
from MDAnalysis.coordinates.memory import MemoryReader



print('Loading pyEmma version: ' + pyemma.__version__)


########################################################################################################################################


#################
#pyEMMA standard Functions
#################

#################
def save_figure(name):
    # change these if wanted
    do_save = True
    fig_dir = './figures/'
    if do_save:
        savefig(fig_dir + name, bbox_inches='tight')

#################
def plot_sampled_function(xall, yall, zall, ax=None, nbins=100, nlevels=20, cmap=cm.bwr, cbar=True, cbar_label=None):
    # histogram data
    xmin = np.min(xall)
    xmax = np.max(xall)
    dx = (xmax - xmin) / float(nbins)
    ymin = np.min(yall)
    ymax = np.max(yall)
    dy = (ymax - ymin) / float(nbins)
    # bin data
    #eps = x
    xbins = np.linspace(xmin - 0.5*dx, xmax + 0.5*dx, num=nbins)
    ybins = np.linspace(ymin - 0.5*dy, ymax + 0.5*dy, num=nbins)
    xI = np.digitize(xall, xbins)
    yI = np.digitize(yall, ybins)
    # result
    z = np.zeros((nbins, nbins))
    N = np.zeros((nbins, nbins))
    # average over bins
    for t in range(len(xall)):
        z[xI[t], yI[t]] += zall[t]
        N[xI[t], yI[t]] += 1.0
    with warnings.catch_warnings() as cm:
        warnings.simplefilter('ignore')
        z /= N
    # do a contour plot
    extent = [xmin, xmax, ymin, ymax]
    if ax is None:
        ax = gca()
    ax.contourf(z.T, 100, extent=extent, cmap=cmap)
    if cbar:
        cbar = plt.colorbar()
        if cbar_label is not None:
            cbar.ax.set_ylabel(cbar_label)
            
    return ax

#################
def plot_sampled_density(xall, yall, zall, ax=None, nbins=100, cmap=cm.Blues, cbar=True, cbar_label=None):
    return plot_sampled_function(xall, yall, zall, ax=ax, nbins=nbins, cmap=cmap, cbar=cbar, cbar_label=cbar_label)




########################################################################################################################################

#################
#MSM In-House Functions 
#################




#################
def plot_projected_density(Z, zall, plt, xdim, ydim, labels, nbins=100, nlevels=20, cmap=cm.bwr, cbar=False, cbar_label=None):
    
    if cbar:
        fig, ax = plt.subplots(figsize=(12,9))
    else:
        fig, ax = plt.subplots(figsize=(9,9))
    
    xall=Z[:,xdim]
    yall=Z[:,ydim]
    
    # histogram data
    xmin = np.min(xall)
    xmax = np.max(xall)
    dx = (xmax - xmin) / float(nbins)
    ymin = np.min(yall)
    ymax = np.max(yall)
    dy = (ymax - ymin) / float(nbins)
    # bin data
    #eps = x
    xbins = np.linspace(xmin - 0.5*dx, xmax + 0.5*dx, num=nbins)
    ybins = np.linspace(ymin - 0.5*dy, ymax + 0.5*dy, num=nbins)
    xI = np.digitize(xall, xbins)
    yI = np.digitize(yall, ybins)
    # result
    z = np.zeros((nbins, nbins))
    N = np.zeros((nbins, nbins))
    # average over bins
    for t in range(len(xall)):
        z[xI[t], yI[t]] += zall[t]
        N[xI[t], yI[t]] += 1.0
    with warnings.catch_warnings() as cm:
        warnings.simplefilter('ignore')
        z /= N
    # do a contour plot
    extent = [xmin, xmax, ymin, ymax]

    lev_step=0.0001
    lev=[x*lev_step for x in range(400)]
    plt.contourf(z.T, 100, extent=extent, cmap=cmap, levels = lev)
    plt.clim(0,0.05)
    
    plt.rc('text', usetex=True)
    
    if cbar:
        cbar = plt.colorbar()
        if cbar_label is not None:
            cbar.ax.set_ylabel(cbar_label, rotation=90, fontsize=30)
            cbar.ax.tick_params(labelsize=30)

    plt.xlim(xbins.min()-5,xbins.max()+5)
    plt.ylim(ybins.min()-5,ybins.max()+5)
    plt.xticks(fontsize=30, rotation=0)
    plt.yticks(fontsize=30, rotation=0)
    plt.xlabel(labels[xdim], fontsize=30)
    plt.ylabel(labels[ydim], fontsize=30)
    
    return plt

#################
#################
def plot_metastable_sets(plt,cl_obj,meta_sets,MSM_dims,dim,mstate_color,msize=10,annotate=False,textsize=12):
    for k in range(len(meta_sets)):
        cl_x=cl_obj.clustercenters[meta_sets[k],np.where(MSM_dims==dim[0])[0][0]]
        cl_y=cl_obj.clustercenters[meta_sets[k],np.where(MSM_dims==dim[1])[0][0]]
        plt.plot(cl_x,cl_y, linewidth=0, marker='o', markersize=msize, markeredgecolor=mstate_color[k],markerfacecolor=mstate_color[k], markeredgewidth=2)
        #plot(cl_obj.clustercenters[meta_sets[k],np.where(MSM_dims==dim[0])[0][0]],cl_obj.clustercenters[meta_sets[k],np.where(MSM_dims==dim[1])[0][0]], linewidth=0, marker='o', markersize=msize, markeredgecolor=mstate_color[k],markerfacecolor=mstate_color[k], markeredgewidth=2)

        if annotate is True:
            plt=annotate_microstates(plt,meta_sets[k],cl_x,cl_y,tsize=textsize)
        
    return 


#################
#Lambda coordinate space
def lambda_obj(lamdir,sims_array,num_frames=None):
    """
    # loads values from lambda space into lambda_obj
    """ 
    coords=[]
    for i in range(len(sims_array)):
        filename=lamdir + '/' + str(sims_array[i][0])+'-'+str(sims_array[i][1]) + '.dat'
        if os.path.isfile(filename):
            tmpcoords=read_float_matrix(filename)
            if num_frames==None:
                coords.append(tmpcoords[:,3:6])
            else:
                coords.append(tmpcoords[0:num_frames,3:6])
    return coords

#################
#Lambda coordinate space
def feature_obj(Z,num_frames=None):
    """
    # loads values from existing numpy array into feature_obj
    """ 
    coords=[]
    if num_frames==None:
        coords.append(Z)
    else:
        coords.append(Z[0:num_frames,:])
    return coords

#################

#################
#Plot Timescale curves
def plot_its(mplt,matplotlib,its_dim_type,x_lim,y_lim):
    #Plot relaxation timescales
    matplotlib.rcParams.update({'font.size': 20})
    mplt.plot_implied_timescales(its_dim_type, ylog=True, dt=0.01, units='ns', linewidth=2)
    #xlim(0, x_lim); 
    #ylim(0, y_lim);
    #save_figure('its.png')
    
    return

#################    
def plot_timescale_ratios(its,ntims=5,ylim=4):

    tim=np.transpose(its.timescales)
    lags=its.lags
    fig, ax = plt.subplots(figsize=(6,4))
    for i in range(ntims):
        plt.plot(lags/10,tim[i]/tim[i+1],'-o',label="$t_{"+str(i+1)+"}$/$t_{"+str(i+2)+"}$")

    plt.rc('text', usetex=True)    
    
    plt.xlim(0,30+np.max(lags)/10)
    plt.ylim(0,ylim)
    plt.xticks(fontsize=30, rotation=0)
    plt.yticks(fontsize=30, rotation=0)
    plt.xlabel("Time (ns)", fontsize=30)
    plt.ylabel(r"$t_{i}/t_{i+1}$", fontsize=30)

    legend = plt.legend(loc='upper right', shadow=False, fontsize='small')

    return    

#################
def plot_kinetic_variance(its,ylim=20):

    lags=its.lags
    fig, ax = plt.subplots(figsize=(6,4))
    kinvar=[(M.eigenvalues()**2).sum() for M in its.models]
    plt.plot(0.1*lags, kinvar, linewidth=2)
    

    plt.rc('text', usetex=True)    
    
    plt.xlim(0,np.max(lags)/10)
    plt.ylim(0,ylim)
    plt.xticks(fontsize=30, rotation=0)
    plt.yticks(fontsize=30, rotation=0)
    plt.xlabel("Time (ns)", fontsize=30)
    plt.ylabel(r"$\sigma^{2}$", fontsize=30)

    return


#################
#File Writing Functions
#################

#################
def write_list_to_file(fname,lname):
    """
    #Writes a list to a filename: fname is filename, lname is list name e.g. traj_list
    """        
    with open(fname,'w') as f:
        for item in lname:
            f.write("%s\n" % item)

    return

#################
def save_current_fig(plt, figname, x, y):

    fig = plt.gcf()
    fig.set_size_inches(x, y)
    plt.savefig(figname, dpi=600, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)
    
    return



#################
def nondiag_rates(kon):
    nd_rates=np.zeros((0,4))
    for i in range(len(kon)):
        for j in range(i+1,len(kon)):
            nd_rates=np.vstack((nd_rates, [int(i), int(j), kon[i,j], kon[j,i]]))
            
    return nd_rates



########################################################################################################################################