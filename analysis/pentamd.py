# PentaMD functions

#Author: S. Kashif Sadoq
#Correspondence: kashif.sadiq@embl.de
#Affiliation: 1. Heidelberg Institute for Theoretical Studies, HITS gGmbH 2. European Moelcular Biology Laboratory

#File Description:
#This file contains a module of python functions for analysing molecular dynamnics (MD) simulations of pentassaccharide systems 

#Citation:
#Balogh, Gabor; Gyöngyösi, Tamás; Timári, István; Herczeg, Mihály; Borbás, Anikó; Sadiq, S. Kashif; Fehér, Krisztina; Kövér, Katalin, 
#Conformational Analysis of Heparin-Analogue Pentasaccharides by Nuclear Magnetic Resonance Spectroscopy and Molecular Dynamics Simulations (2021), 
#Journal of Chemical Information and Modeling, Accepted.

#Last modified: 13/5/2021

########################################################################################################################################

# Module dependencies

import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import gridspec
from collections import OrderedDict


import MDAnalysis as mda
from MDAnalysis.analysis import dihedrals
from MDAnalysis.analysis import align, rms, distances, contacts
from MDAnalysis.analysis.base import AnalysisFromFunction 
from MDAnalysis.coordinates.memory import MemoryReader


from sklearn.neighbors import KernelDensity
from scipy.stats import norm


########################################################################################################################################

#################
#File reading functions
#################
def read_int_matrix(fname):
    """
    reads a file containing a matrix of integer numbers
    """
    a = []
    with open(fname) as f:
        for line in f:
            row = line.rstrip().split()
            a.append(row)
    foo = np.array(a)
    bar = foo.astype(np.int)
    return bar

#################
#Read in matrix of floats from file
def read_float_matrix(fname):
    """
    reads a file containing a matrix of floating point numbers
    """
    a = []
    with open(fname) as f:
        for line in f:
            row = line.rstrip().split()
            a.append(row)
    foo = np.array(a)
    bar = foo.astype(np.float)
    return bar


def READ_INITIAL_FILE ( filename ):
    # read in group data into lists of lists
    
    file = open(filename,'r')
    coords=[]
    for line in file:
        vals=line.split()
        vals2 = [float(numeric_string) for numeric_string in vals[3:6]]        
        coords.append(vals2)
        
    return coords;




########################################################################################################################################

# Loading MD trajectories 

def traj_dir(path,sys):
    
    traj_directory=str(path)+str(sys)+'/filtered/'    
    
    return traj_directory

def traj_concatenate(path,sys):

    traj_directory=traj_dir(path,sys)
    traj_concat=[]
    for i in range(1,11):
        traj_concat.append(traj_directory+str(i)+'_md.nc')

    return traj_concat


def U_ensemble(path,sys):

    U = []
    for i in range(1,11):
        u = mda.Universe(traj_dir(path,sys)+'complex_nowat.pdb',traj_dir(path,sys)+str(i)+'_md.nc')
        U.append(u)

    return U

########################################################################################################################################

# Hydrogen-Hydrogen distance functions

def h1_h2_atom_select(u,resid_dict, nuclei_list, pair_id):

    resid1=resid_dict[nuclei_list[pair_id][0][0]]
    h_num1=nuclei_list[pair_id][0][1:]
    resid2=resid_dict[nuclei_list[pair_id][1][0]]
    h_num2=nuclei_list[pair_id][1][1:]
    H1=u.select_atoms('resid '+str(resid1)+' and name H'+str(h_num1))
    H2=u.select_atoms('resid '+str(resid2)+' and name H'+str(h_num2))

    return H1, H2

def pair_id_avg_dist(U,resid_dict, nuclei_list, pair_id):
    
    pos=np.empty((0,1))
    for j in range(len(U)):
        u=U[j]
        H1,H2=h1_h2_atom_select(u,resid_dict, nuclei_list, pair_id)
        for ts in u.trajectory[0::10]:
            pos=np.vstack((pos,np.linalg.norm(H1.positions-H2.positions) ))

    dist=np.array((np.mean(pos), np.std(pos)))
    
    return dist



def nuclei_list_avg_dist(U,resid_dict, nuclei_list):
    
    dist=np.empty((0,2))
    for pair_id in range(len(nuclei_list)):
        dist=np.vstack((dist,pair_id_avg_dist(U,resid_dict, nuclei_list, pair_id) ))

    return dist


########################################################################################################################################

# Distribution functions

def kde_function(Y,Y_plot,h):
    
    Y_range=Y_plot.reshape((-1,1))    
    kde = KernelDensity(kernel='epanechnikov', bandwidth=h).fit(Y)
    log_dens = kde.score_samples(Y_range)
    
    return Y_range[:,0], np.exp(log_dens)


def timeseries_axes(ax,xlim,ylim,x_ticks,y_ticks,xlabel,ylabel):
    
    ax.set_xlim(xlim[0],xlim[1])
    ax.set_xlabel(xlabel,fontsize=30)
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='x', labelsize=30)

    ax.set_ylim(ylim[0],ylim[1])
    ax.set_ylabel(ylabel,fontsize=30)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis='y', labelsize=30)

    return ax


def distribution_plot(gs_id,data,D_space,h,color,alph=1):

    ax = plt.subplot(gs_id)
    D=data.reshape(-1,1)
    yfunc, epan = kde_function(D,D_space,h)    
    ax.fill(yfunc,epan, fc=color,alpha=alph)

    return ax



########################################################################################################################################

# RMSD functions

def rmsd_analysis(path,sys,rep,ref_prmtop, ref_pdb,selection):

    ALIGN_SELECTION=selection

    trajfile=traj_dir(path,sys)+str(rep)+'_md.nc'
    u_ref = mda.Universe(ref_prmtop,ref_pdb)
    u = mda.Universe(traj_dir(path,sys)+'complex_nowat.pdb',trajfile)

    #RMSD
    R = rms.RMSD(u, u_ref, select=ALIGN_SELECTION, groupselections=[],ref_frame=0)
    R.run()

    return R.rmsd

def ensemble_rmsd(path,sys,ref_prmtop,ref_pdb,selection):

    for i in range(10):
        rep=i+1
        RMS=rmsd_analysis(path,sys,rep,ref_prmtop, ref_pdb,selection)
    
        if i==0:
            Rall=RMS
        else:
            Rall=np.hstack((Rall,RMS[:,2].reshape(-1,1)))

    return Rall


def plot_rmsd_panel(plt,panel,data,col,dat_type="rmsd",xlab=True,xax=True,ylab=True,yax=True,LS='',LCOL='k',M='o',msize=2):
    """
    Plotting RMSD of pentasaccharide backbone
    """
    
    xlim=[0,2000]
    x_ticks=[x for x in range(0,2000,1000)]
    
    if dat_type=="rmsd":
        ylim=[0,3]
        y_ticks=[y for y in range(0,3)]
        Ylabel=r"$RMSD$ (\AA)"
    
    ax = plt.subplot(panel)
    ax.plot([x/100 for x in range(0,len(data),1000)], data[0::1000,col], ls=LS,color=LCOL,marker=M,ms=msize)

    ax.set_xlim(xlim[0],xlim[1])
    ax.set_ylim(ylim[0],ylim[1])
    
    #Labels
    if xlab==True:
        ax.set_xlabel("Time (ns)",fontsize=20)
    if ylab==True:
        ax.set_ylabel(Ylabel,fontsize=20)
    
    #Ticks

    if xax==True:        
        ax.set_xticks(x_ticks)
        ax.tick_params(axis='x', labelsize=20)
    else:
        ax.set_xticks([])
    if yax==True:
        ax.set_yticks(y_ticks)
        ax.tick_params(axis='y', labelsize=20)
    else:
        ax.set_yticks([])


def plot_rmsd_timeseries(plt,allR):
    
    fig = plt.figure(figsize=(18, 8)) 
    plt.rc('text', usetex=True) 
    fig.subplots_adjust(hspace=1.0, wspace=0.05)
    gs = gridspec.GridSpec(3, 10, width_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                       height_ratios=[1,1,1])


    gs_base=0
    dat_type="rmsd"



    for j in range(3):
        data=allR[j]

        for i in range(10):
            if i == 0:
                plot_rmsd_panel(plt,gs[i+gs_base],data,i+2,dat_type,1,1,1,1,LCOL='k')
            else:
                plot_rmsd_panel(plt,gs[i+gs_base],data,i+2,dat_type,1,1,0,0,LCOL='k')

        gs_base+=10

    
    fig.text(0.01, 0.84, '1', fontsize=50, color='k')
    fig.text(0.01, 0.49, '2', fontsize=50, color='k')
    fig.text(0.01, 0.14, '3', fontsize=50, color='k')

    fig.text(0.16, 0.9, '1', fontsize=20, color='k')
    fig.text(0.24, 0.9, '2', fontsize=20, color='k')
    fig.text(0.32, 0.9, '3', fontsize=20, color='k')
    fig.text(0.39, 0.9, '4', fontsize=20, color='k')
    fig.text(0.47, 0.9, '5', fontsize=20, color='k')
    fig.text(0.54, 0.9, '6', fontsize=20, color='k')
    fig.text(0.62, 0.9, '7', fontsize=20, color='k')
    fig.text(0.70, 0.9, '8', fontsize=20, color='k')
    fig.text(0.78, 0.9, '9', fontsize=20, color='k')
    fig.text(0.86, 0.9, '10', fontsize=20, color='k')


    return fig,plt


def plot_rmsd_distribution(plt,allR):

    fig = plt.figure(figsize=(18, 4)) 
    fig.subplots_adjust(hspace=0.1, wspace=0.5)
    gs = gridspec.GridSpec(1, 3)     
        
    h=0.2
    D_space=np.array([y/100 for y in range(0,300)])
    color='k'
    alph=1
    x_range=[0,3]
    x_ticks=[x for x in range(0,3,1)]
    y_range=[0,1.5]
    y_ticks=[0.5,1,1.5]
    Xlabel=r"$RMSD$ (\AA)"
    Ylabel=r"$\rho$"

    ax=distribution_plot(gs[0],allR[0][::100,2:],D_space,h,color,alph=alph)
    ax=timeseries_axes(ax,x_range,y_range,x_ticks,y_ticks,Xlabel,Ylabel)

    ax=distribution_plot(gs[1],allR[1][::100,2:],D_space,h,color,alph=alph)
    ax=timeseries_axes(ax,x_range,y_range,x_ticks,y_ticks,Xlabel,Ylabel)

    ax=distribution_plot(gs[2],allR[2][::100,2:],D_space,h,color,alph=alph)
    ax=timeseries_axes(ax,x_range,y_range,x_ticks,y_ticks,Xlabel,Ylabel)


    fig.text(0.20, 0.95, '1', fontsize=40, color='k')
    fig.text(0.5, 0.95, '2', fontsize=40, color='k')
    fig.text(0.8, 0.95, '3', fontsize=40, color='k')

    return fig,plt
        
########################################################################################################################################

# Ring puckering and glycosidic linkage functions

def atoms_str(atoms_list, ang_type=0):
    #ang_type=0: phi, ang_type=1: psi
    atoms_str=""
    for i in range(4):
        atoms_str = atoms_str + str(atoms_list[i+ang_type])
        if i < 3:
            atoms_str = atoms_str + " or "

    return atoms_str

def dihed_atoms_str(dihed_atoms_list):
    atoms_str=""
    for i in range(4):
        atoms_str = atoms_str + str(dihed_atoms_list[i])
        if i < 3:
            atoms_str = atoms_str + " or "

    return atoms_str


def dlist(universe, glink, ring_ordering):
    diheds_list=[]
    for ring in range(5,1,-1):
        a=[]
        for i in range(len(glink)):
            a.append("(resid " + str(ring+ring_ordering[i]) + " and name " + str(glink[i])+ ")" )
    
        phi_atoms_str=dihed_atoms_str(a[0:4])
        psi_atoms_str=dihed_atoms_str(a[1:5])
        
        #phi_atoms_str=atoms_str(a,0)
        #psi_atoms_str=atoms_str(a,1)

        phi=universe.select_atoms(phi_atoms_str)
        psi=universe.select_atoms(psi_atoms_str)

        diheds_list.append(phi)
        diheds_list.append(psi)

    return diheds_list


def angle_wrap(data,shift):
    outdata_array=np.empty((0,1))
    #outdata=[]
    for i in range(len(data)):        
        outdata = data[i] + shift
        
        if outdata < -180:
            outdata=outdata+360
        elif outdata > 180:
            outdata=outdata-360
    
        
        outdata_array=np.vstack((outdata_array,outdata))
    
    #outdata_array=np.transpose(outdata_array)
            
    return outdata_array


def gl_phipsi_array(dihs):
    """
    Makes array of phipsi glycosidic linkage dihedrals across all links between the rings
    Shifts angle by -180 for every psi linkage as just a way of reporting it with angle lablelling from the other direction
    """
    gl_phipsi=np.empty((len(dihs.angles[:,0]),0))
    for i in range(8):
        if (-1)**i==1:
            gl_phipsi=np.hstack((gl_phipsi,angle_wrap(dihs.angles[:,i],0)))
        elif (-1)**i==-1:
            gl_phipsi=np.hstack((gl_phipsi,angle_wrap(dihs.angles[:,i],-180)))
    
    return gl_phipsi

#Fix for G/H psi wrap 
def fix_gl(gl):
    """
    Fixes the wrapping problem for last column of gl data
    """
    for i in range(len(gl)):
        if gl[i,7]<-100:
            gl[i,7] += 360
    
    return gl


def glycosidic_link_dihedral_angles(u,glink,ring_ordering):
    
    diheds_list=dlist(u, glink, ring_ordering)
    dihs = dihedrals.Dihedral(diheds_list).run()
    gl=gl_phipsi_array(dihs)
    #gl=fix_gl(gl)

    return gl

########################################################################################################################################

# Cremer-Pople functions

def cremer_pople_hex(u, resid, ring_atoms):
    """
    Calculates Q,theta, phi Cremer-Pole parameters from a list containing all 6 atom names in the hexameric ring
    On a single frame
    In-house developed code version 
    """
    N=len(ring_atoms)
    #sqrt_2 = math.sqrt(2.)
    #inv_sqrt_6 = math.sqrt(1./N)

    R=np.empty((0,3))
    for x in range(len(ring_atoms)):
        R=np.vstack((R,u.select_atoms("resid " + str(resid) + " and name " + str(ring_atoms[x])).positions))
    R_cog=np.sum(R,axis=0)/N
    R0=R-R_cog
    
    R1 = np.sum(np.array([R0[j]*np.sin(2*pi*j/N) for j in range(len(R0))]),axis=0)
    R2 = np.sum(np.array([R0[j]*np.cos(2*pi*j/N) for j in range(len(R0))]),axis=0)
    n = np.cross(R1,R2)
    n=n/np.linalg.norm(n)
    z=np.dot(R0,n)

    Q=np.sqrt(np.sum(z**2))

    q2cos = np.sqrt(2/N)*np.sum(np.array([z[j]*cos(4*pi*j/N) for j in range(len(z))]))
    q2sin = -1*np.sqrt(2/N)*np.sum(np.array([z[j]*sin(4*pi*j/N) for j in range(len(z))]))
    
    #q2cos = sqrt_2*inv_sqrt_6*np.sum(np.array([z[j]*cos(4*pi*j/N) for j in range(len(z))]))
    #q2sin = -1*sqrt_2*inv_sqrt_6*np.sum(np.array([z[j]*sin(4*pi*j/N) for j in range(len(z))]))
    
    q2=np.sqrt(q2cos**2+q2sin**2)
    phi=np.degrees(np.arctan2(q2sin,q2cos)) % 360
    
    q3=np.sqrt(1/N)*np.sum(np.array([((-1)**j)*z[j] for j in range(len(z))]))
    #q3=inv_sqrt_6*np.sum(np.array([((-1)**j)*z[j] for j in range(len(z))]))
    theta=np.degrees(np.arccos(q3/Q))
    
    #return Q,theta,phi
    return theta,phi


def cremer_pople_analysis(u,resids):
    """
    Cremer-Pople analysis across a trajectory  specifying which hex ring resids to compute
    """
    ring=["O5", "C1", "C2", "C3", "C4", "C5"]
    cp = np.empty((0,np.int(len(resids)*3)))
    for ts in u.trajectory:
        for res in resids:
            if res == 5:
                vals=cremer_pople_hex(u, res, ring)
            else:
                vals=np.hstack((vals,cremer_pople_hex(u, res, ring)))
    
        cp = np.vstack((cp,vals))    
        
    return cp


########################################################################################################################################

# Functions for Loading Precomputed CP and GL Data 

def load_data_array(analog=1,rep_start=1,rep_end=1,data_type='cp'):

    for i in range(rep_start-1,rep_end):
        repnum=i+1
        filename='./data/'+str(data_type)+'_'+str(analog)+'_'+str(repnum)+'.dat'
        if i==0:
            data_array=read_float_matrix(filename)
        else:
            data_array=np.dstack((data_array,read_float_matrix(filename)))
        
    return data_array



def join_in_Y_format(CP,GL):
    # Join All feature data in Y format for pyemma

    Y=[]
    for i in range(np.shape(CP)[2]):
        Y.append(np.hstack((CP[:,:,i],GL[:,:,i])))

    return Y


def concat_replicas(Y):
    
    Z=np.reshape(Y,(np.shape(Y)[0]*np.shape(Y)[1],np.shape(Y)[2]))
    
    
    return Z

########################################################################################################################################


#  Generaral 2D density/Free energy/PMF landscape Functions


#################
def plot_weighted_free_energy_landscape(Z,plt,xdim,ydim,labels, cmap="jet", fill=True, 
                                        contour_label=True, contour_color='k', clim=[-10,0],cbar=False, 
                                        cbar_label="G (kcal/mol)",lev_max=-1,shallow=False,wg=None,
                                        fsize_cbar=(12,9),fsize=(9,9)):
    
    if cbar:
        fig, ax = plt.subplots(figsize=fsize_cbar)
    else:
        fig, ax = plt.subplots(figsize=fsize)
    

    #x=np.vstack(Y)[:,0]
    #y=np.vstack(Y)[:,2]
    x=Z[:,xdim]
    y=Z[:,ydim]
    rho,xbins,ybins = np.histogram2d(x,y,bins=[100,100],weights=wg)
    rho += 0.1
    kBT=0.596
    G=-kBT*np.log(rho/np.sum(rho))
    G=G-np.max(G)
    ex=[xbins.min(),xbins.max(),ybins.min(),ybins.max()]
    lev=[x /10.0 for x in range(int(5*round(np.min(G)*2))-10,int(lev_max*10),5)]
    if shallow is True:
        lev_shallow=[-0.4,-0.3,-0.2,-0.1]
        lev+=lev_shallow    
    contours=plt.contour(np.transpose(G), extent=ex, levels = lev, colors=contour_color, linestyles= '-' )
    
    if fill is True:
        plt.contourf(np.transpose(G), extent=ex,cmap = cmap, levels = lev)
    
    if contour_label is True:
        plt.clabel(contours, inline=True, fmt='%1.1f', fontsize=20)
    
    plt.clim(clim[0],clim[1])
    
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


def plot_weighted_free_energy_landscape_array(Z,plt,xdim,ydim,labels, cmap="jet", fill=True, 
                                              contour_label=True, contour_color='k', clim=[-10,0],cbar=False, 
                                              cbar_label="G (kcal/mol)",lev_max=-1,shallow=False,wg=None,
                                              fixrange=True, x_lim=[0,360],y_lim=[0,180],x_ticksep=30,y_ticksep=30,
                                              xlabels=True,ylabels=True,xticks=True,yticks=True):

    #x=np.vstack(Y)[:,0]
    #y=np.vstack(Y)[:,2]
    x=Z[:,xdim]
    y=Z[:,ydim]
    rho,xbins,ybins = np.histogram2d(x,y,bins=[100,100],weights=wg)
    rho += 0.1
    kBT=0.596
    G=-kBT*np.log(rho/np.sum(rho))
    G=G-np.max(G)
    ex=[xbins.min(),xbins.max(),ybins.min(),ybins.max()]
    lev=[x /10.0 for x in range(int(5*round(np.min(G)*2))-10,int(lev_max*10),5)]
    if shallow is True:
        lev_shallow=[-0.4,-0.3,-0.2,-0.1]
        lev+=lev_shallow    
    contours=plt.contour(np.transpose(G), extent=ex, levels = lev, colors=contour_color, linestyles= '-' )
    
    if fill is True:
        plt.contourf(np.transpose(G), extent=ex,cmap = cmap, levels = lev)
    
    if contour_label is True:
        plt.clabel(contours, inline=True, fmt='%1.1f', fontsize=20)
    
    plt.clim(clim[0],clim[1])
    
    plt.rc('text', usetex=True)
        
    if cbar:
        cbar = plt.colorbar()
        if cbar_label is not None:
            cbar.ax.set_ylabel(cbar_label, rotation=90, fontsize=30)
            cbar.ax.tick_params(labelsize=30)    
    
    if fixrange==True:
        plt.xlim(x_lim[0],x_lim[1])
        plt.ylim(y_lim[0],y_lim[1])
        x_ticks=[x for x in range(x_lim[0],x_lim[1],x_ticksep)]
        y_ticks=[y for y in range(y_lim[0],y_lim[1],y_ticksep)]
        
        plt.xticks(x_ticks)
        plt.yticks(y_ticks)
    
    else:
        plt.xlim(xbins.min()-5,xbins.max()+5)
        plt.ylim(ybins.min()-5,ybins.max()+5)
    
    if xticks:            
        plt.xticks(fontsize=20, rotation=0)
    
    
    if yticks:
        plt.yticks(fontsize=20, rotation=0)
    else:
        plt.yticks(y_ticks,'',fontsize=20, rotation=0)
    
    if xlabels:
        plt.xlabel(labels[xdim], fontsize=20)
    if ylabels:
        plt.ylabel(labels[ydim], fontsize=20)
    
    
    return plt


#################
def plot_weighted_free_energy_landscape_range(Z,plt,xdim,ydim,labels, cmap="jet", fill=True,  
                                              contour_label=True, contour_color='k', clim=[-10,0],cbar=False, 
                                              cbar_label="G (kcal/mol)",lev_max=-1,shallow=False,wg=None,
                                              fsize_cbar=(12,9),fsize=(9,9),
                                              fixrange=True,x_lim=[0,360],y_lim=[0,180],x_ticksep=30,y_ticksep=30):

    if cbar:
        fig, ax = plt.subplots(figsize=fsize_cbar)
    else:
        fig, ax = plt.subplots(figsize=fsize)
    

    #x=np.vstack(Y)[:,0]
    #y=np.vstack(Y)[:,2]
    x=Z[:,xdim]
    y=Z[:,ydim]
    rho,xbins,ybins = np.histogram2d(x,y,bins=[100,100],weights=wg)
    rho += 0.1
    kBT=0.596
    G=-kBT*np.log(rho/np.sum(rho))
    G=G-np.max(G)
    ex=[xbins.min(),xbins.max(),ybins.min(),ybins.max()]
    lev=[x /10.0 for x in range(int(5*round(np.min(G)*2))-10,int(lev_max*10),5)]
    if shallow is True:
        lev_shallow=[-0.4,-0.3,-0.2,-0.1]
        lev+=lev_shallow    
    contours=plt.contour(np.transpose(G), extent=ex, levels = lev, colors=contour_color, linestyles= '-' )
    
    if fill is True:
        plt.contourf(np.transpose(G), extent=ex,cmap = cmap, levels = lev)
    
    if contour_label is True:
        plt.clabel(contours, inline=True, fmt='%1.1f', fontsize=20)
    
    plt.clim(clim[0],clim[1])
    
    plt.rc('text', usetex=True)
        
    if cbar:
        cbar = plt.colorbar()
        if cbar_label is not None:
            cbar.ax.set_ylabel(cbar_label, rotation=90, fontsize=30)
            cbar.ax.tick_params(labelsize=30)    
    
    if fixrange==True:
        plt.xlim(x_lim[0],x_lim[1])
        plt.ylim(y_lim[0],y_lim[1])
        x_ticks=[x for x in range(x_lim[0],x_lim[1],x_ticksep)]
        y_ticks=[y for y in range(y_lim[0],y_lim[1],y_ticksep)]
        plt.xticks(x_ticks)
        plt.yticks(y_ticks)
    else:
        plt.xlim(xbins.min()-5,xbins.max()+5)
        plt.ylim(ybins.min()-5,ybins.max()+5)
    
    plt.xticks(fontsize=30, rotation=0)
    plt.yticks(fontsize=30, rotation=0)         
    plt.xlabel(labels[xdim], fontsize=40)
    plt.ylabel(labels[ydim], fontsize=40)
    
    return plt


#################

########################################################################################################################################

#  CP and GL Data Plotting Functions

#################
def plot_single_trajectory(plt,data):

    fig, ax = plt.subplots(figsize=(8,4))
    plt.plot([x/100 for x in range(np.shape(data)[0])],data,'.k')
    plt.rc('text', usetex=True)    

    x_ticks=[x for x in range(0,2001,500)]
    y_ticks=[y for y in range(-180,181,90)]

    plt.xlim(0,2000)
    plt.ylim(-180,180)
    plt.xticks(x_ticks, fontsize=30, rotation=0)
    plt.yticks(y_ticks,fontsize=30, rotation=0)
    plt.xlabel("Time (ns)", fontsize=30)
    plt.ylabel(r"$\phi$", fontsize=30)
    
    return fig,plt


#################
def plot_panel(plt,panel,data,col,dat_type="tcp",xlab=True,xax=True,ylab=True,yax=True,LS='',LCOL='k',M='o',msize=2):
    """
    Plotting Cremer-Pople and  glycosidic linkage angles
    """
    
    xlim=[0,2000]
    x_ticks=[x for x in range(0,2000,1000)]
    
    if dat_type=="qcp":
        ylim=[0,1]
        y_ticks=[y/10 for y in range(0,10,5)]
        Ylabel=r"$Q_{cp}$"
    elif dat_type=="tcp":
        ylim=[0,180]
        y_ticks=[y for y in range(0,181,90)]
        Ylabel=r"$\theta_{cp}$ ($^{\circ}$)"
    elif dat_type=="pcp":
        ylim=[0,360]
        y_ticks=[y for y in range(0,361,180)]
        Ylabel=r"$\phi_{cp}$ ($^{\circ}$)"
    elif dat_type=="phi":
        ylim=[-180,180]
        y_ticks=[y for y in range(-90,180,90)]
        Ylabel=r"$\phi_{gl}$ ($^{\circ}$)"
    elif dat_type=="psi":
        ylim=[-180,180]
        y_ticks=[y for y in range(-90,180,90)]
        Ylabel=r"$\psi_{gl}$ ($^{\circ}$)"
        
    #Ylabels=[r"$Q_{cp}$", r"$\theta_{cp}$ ($^{\circ}$)", r"$\phi_{cp}$ ($^{\circ}$)"]
    
    ax = plt.subplot(panel)
    ax.plot([x/100 for x in range(0,len(data),1000)], data[0::1000,col], ls=LS,color=LCOL,marker=M,ms=msize)

    ax.set_xlim(xlim[0],xlim[1])
    ax.set_ylim(ylim[0],ylim[1])
    
    #Labels
    if xlab==True:
        ax.set_xlabel("Time (ns)",fontsize=20)
    if ylab==True:
        ax.set_ylabel(Ylabel,fontsize=20)
    
    #Ticks

    if xax==True:        
        ax.set_xticks(x_ticks)
        ax.tick_params(axis='x', labelsize=20)
    else:
        ax.set_xticks([])
    if yax==True:
        ax.set_yticks(y_ticks)
        ax.tick_params(axis='y', labelsize=20)
    else:
        ax.set_yticks([])
    
    return
    
#################
def plot_cp_gl_timeseries(plt,CP,GL):
    

    fig = plt.figure(figsize=(18, 30)) 
    plt.rc('text', usetex=True) 
    fig.subplots_adjust(hspace=0.2, wspace=0.05)

    gs = gridspec.GridSpec(18, 10, width_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                           height_ratios=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])


    gs_base=0

    for r in range(4):
        ring_id=r
        gl_id=ring_id

        D_COL=[2*ring_id,2*ring_id+1,2*gl_id,2*gl_id+1]
        D_TYPE=["tcp","pcp","phi","psi"]
        D_LINECOL=['k','k','#b2182b','#2166ac']
        DAT=[CP,CP,GL,GL]


        for d in range(len(D_COL)):
            data=DAT[d]
            dcol=D_COL[d]
            dat_type=D_TYPE[d]
            dat_linecol=D_LINECOL[d]
    
    
            for i in range(10):
                if i == 0:
                    plot_panel(plt,gs[i+gs_base],data[:,:,i],dcol,dat_type,0,0,1,1,LCOL=dat_linecol)
                else:
                    plot_panel(plt,gs[i+gs_base],data[:,:,i],dcol,dat_type,0,0,0,0,LCOL=dat_linecol)
            gs_base+=10        

        
    ring_id=4
    gl_id=ring_id

    D_COL=[2*ring_id,2*ring_id+1]
    D_TYPE=["tcp","pcp"]
    D_LINECOL=['k','k']
    DAT=[CP,CP]


    for d in range(len(D_COL)):
        data=DAT[d]
        dcol=D_COL[d]
        dat_type=D_TYPE[d]
        dat_linecol=D_LINECOL[d]
    
    
        if d == 0:
    
            for i in range(10):
                if i == 0:
                    plot_panel(plt,gs[i+gs_base],data[:,:,i],dcol,dat_type,0,0,1,1,LCOL=dat_linecol)
                else:
                    plot_panel(plt,gs[i+gs_base],data[:,:,i],dcol,dat_type,0,0,0,0,LCOL=dat_linecol)
            
        else:
        
            for i in range(10):
                if i == 0:
                    plot_panel(plt,gs[i+gs_base],data[:,:,i],dcol,dat_type,1,1,1,1,LCOL=dat_linecol)
                else:
                    plot_panel(plt,gs[i+gs_base],data[:,:,i],dcol,dat_type,1,1,0,0,LCOL=dat_linecol)
        
            
        gs_base+=10        
    
    
    fig.text(0.01, 0.84, 'D', fontsize=50, color='k')
    fig.text(0.01, 0.67, 'E', fontsize=50, color='k')
    fig.text(0.01, 0.49, 'F', fontsize=50, color='k')
    fig.text(0.01, 0.32, 'G', fontsize=50, color='k')
    fig.text(0.01, 0.14, 'H', fontsize=50, color='k')

    fig.text(0.01, 0.76, 'D/E', fontsize=30, color='k')
    fig.text(0.01, 0.59, 'E/F', fontsize=30, color='k')
    fig.text(0.01, 0.41, 'F/G', fontsize=30, color='k')
    fig.text(0.01, 0.24, 'G/H', fontsize=30, color='k')   



    fig.text(0.16, 0.9, '1', fontsize=20, color='k')
    fig.text(0.24, 0.9, '2', fontsize=20, color='k')
    fig.text(0.32, 0.9, '3', fontsize=20, color='k')
    fig.text(0.39, 0.9, '4', fontsize=20, color='k')
    fig.text(0.47, 0.9, '5', fontsize=20, color='k')
    fig.text(0.54, 0.9, '6', fontsize=20, color='k')
    fig.text(0.62, 0.9, '7', fontsize=20, color='k')
    fig.text(0.70, 0.9, '8', fontsize=20, color='k')
    fig.text(0.78, 0.9, '9', fontsize=20, color='k')
    fig.text(0.86, 0.9, '10', fontsize=20, color='k')

    return fig,plt

#################    
def plot_cp_gl_density2D(plt,Zlabels,all_Z):
    
    fig = plt.figure(figsize=(18, 54)) 
    plt.rc('text', usetex=True) 
    fig.subplots_adjust(hspace=0.2, wspace=0.05)
    gs = gridspec.GridSpec(9, 3,width_ratios=[0.3,0.3,0.4])


    gs_base=0

    for r in range(4):
        ring_id=r
        gl_id=ring_id
    
        # The Cremer-Pople row
        dim=[2*ring_id+1,2*ring_id]
        for i in range(3):
            Z=all_Z[i]
            ax = plt.subplot(gs[i+gs_base])
    
    
            if i==0:
                plt=plot_weighted_free_energy_landscape_array(Z,plt,dim[0],dim[1],
                                              Zlabels,cmap='Spectral',contour_label=False,cbar=False,x_ticksep=60)
            elif i==1:
                plt=plot_weighted_free_energy_landscape_array(Z,plt,dim[0],dim[1],
                                              Zlabels,cmap='Spectral',contour_label=False,cbar=False,x_ticksep=60,
                                              ylabels=False,yticks=False)
            else:
                plt=plot_weighted_free_energy_landscape_array(Z,plt,dim[0],dim[1],
                                              Zlabels,cmap='Spectral',contour_label=False,cbar=True,x_ticksep=60,
                                              cbar_label="-k$_{B}$Tln(p) (kcal/mol)",ylabels=False,yticks=False)  
        gs_base+=3
    
        # The Glycosidic linkage row
        dim=[2*gl_id+10,2*gl_id+11]
        for i in range(3):
            Z=all_Z[i]
            ax = plt.subplot(gs[i+gs_base])
    
    
            if i==0:
                plt=plot_weighted_free_energy_landscape_array(Z,plt,dim[0],dim[1],
                                              Zlabels,cmap='Spectral',contour_label=False,cbar=False,x_ticksep=60,
                                                         y_ticksep=60,x_lim=[-180,180],y_lim=[-180,180])
            elif i==1:
                plt=plot_weighted_free_energy_landscape_array(Z,plt,dim[0],dim[1],
                                              Zlabels,cmap='Spectral',contour_label=False,cbar=False,x_ticksep=60,
                                              y_ticksep=60,ylabels=False,yticks=False,x_lim=[-180,180],y_lim=[-180,180])
            else:
                plt=plot_weighted_free_energy_landscape_array(Z,plt,dim[0],dim[1],
                                                Zlabels,cmap='Spectral',contour_label=False,cbar=True,x_ticksep=60,
                                              y_ticksep=60,cbar_label="-k$_{B}$Tln(p) (kcal/mol)",ylabels=False,yticks=False,
                                                         x_lim=[-180,180],y_lim=[-180,180])  
        gs_base+=3


    
    ring_id=4
    gl_id=ring_id
    dim=[2*ring_id+1,2*ring_id]

    for i in range(3):
        Z=all_Z[i]
        ax = plt.subplot(gs[i+gs_base])
    
    
        if i==0:
            plt=plot_weighted_free_energy_landscape_array(Z,plt,dim[0],dim[1],
                                              Zlabels,cmap='Spectral',contour_label=False,cbar=False,x_ticksep=60)
        elif i==1:
            plt=plot_weighted_free_energy_landscape_array(Z,plt,dim[0],dim[1],
                                              Zlabels,cmap='Spectral',contour_label=False,cbar=False,x_ticksep=60,
                                              ylabels=False,yticks=False)
        else:
            plt=plot_weighted_free_energy_landscape_array(Z,plt,dim[0],dim[1],
                                              Zlabels,cmap='Spectral',contour_label=False,cbar=True,x_ticksep=60,
                                              cbar_label="-k$_{B}$Tln(p) (kcal/mol)",ylabels=False,yticks=False)
    gs_base+=3


    fig.text(0.01, 0.84, 'D', fontsize=50, color='k')
    fig.text(0.01, 0.67, 'E', fontsize=50, color='k')
    fig.text(0.01, 0.49, 'F', fontsize=50, color='k')
    fig.text(0.01, 0.32, 'G', fontsize=50, color='k')
    fig.text(0.01, 0.14, 'H', fontsize=50, color='k')

    fig.text(0.01, 0.76, 'D/E', fontsize=30, color='k')
    fig.text(0.01, 0.59, 'E/F', fontsize=30, color='k')
    fig.text(0.01, 0.41, 'F/G', fontsize=30, color='k')
    fig.text(0.01, 0.24, 'G/H', fontsize=30, color='k')

    fig.text(0.25, 0.9, '1', fontsize=40, color='k')
    fig.text(0.5, 0.9, '2', fontsize=40, color='k')
    fig.text(0.75, 0.9, '3', fontsize=40, color='k')
    
    return fig,plt

#################
def plot_cp_density2D(plt,Zlabels,all_Z):
    
    fig = plt.figure(figsize=(18, 30)) 
    plt.rc('text', usetex=True) 
    fig.subplots_adjust(hspace=0.2, wspace=0.05)
    gs = gridspec.GridSpec(5, 3,width_ratios=[0.3,0.3,0.4])
    
    gs_base=0
    
    for r in range(5):
        ring_id=r
        gl_id=ring_id
    
        # The Cremer-Pople row
        dim=[2*ring_id+1,2*ring_id]
        for i in range(3):
            Z=all_Z[i]
            ax = plt.subplot(gs[i+gs_base])
    
    
            if i==0:
                plt=plot_weighted_free_energy_landscape_array(Z,plt,dim[0],dim[1],
                                              Zlabels,cmap='Spectral',contour_label=False,cbar=False,x_ticksep=60)
            elif i==1:
                plt=plot_weighted_free_energy_landscape_array(Z,plt,dim[0],dim[1],
                                              Zlabels,cmap='Spectral',contour_label=False,cbar=False,x_ticksep=60,
                                              ylabels=False,yticks=False)
            else:
                plt=plot_weighted_free_energy_landscape_array(Z,plt,dim[0],dim[1],
                                              Zlabels,cmap='Spectral',contour_label=False,cbar=True,x_ticksep=60,
                                              cbar_label="-k$_{B}$Tln(p) (kcal/mol)",ylabels=False,yticks=False)  
        gs_base+=3
    

    fig.text(0.01, 0.80, 'D', fontsize=50, color='k')
    fig.text(0.01, 0.65, 'E', fontsize=50, color='k')
    fig.text(0.01, 0.50, 'F', fontsize=50, color='k')
    fig.text(0.01, 0.35, 'G', fontsize=50, color='k')
    fig.text(0.01, 0.20, 'H', fontsize=50, color='k')


    fig.text(0.25, 0.9, '1', fontsize=40, color='k')
    fig.text(0.5, 0.9, '2', fontsize=40, color='k')
    fig.text(0.75, 0.9, '3', fontsize=40, color='k')
    
    return fig,plt

#################
def plot_gl_density2D(plt,Zlabels,all_Z):
    
    fig = plt.figure(figsize=(18, 24)) 
    plt.rc('text', usetex=True) 
    fig.subplots_adjust(hspace=0.2, wspace=0.05)
    gs = gridspec.GridSpec(4, 3,width_ratios=[0.3,0.3,0.4])

    gs_base=0

    for r in range(4):
        ring_id=r
        gl_id=ring_id
    
        # The Glycosidic linkage row
        dim=[2*gl_id+10,2*gl_id+11]
        for i in range(3):
            Z=all_Z[i]
            ax = plt.subplot(gs[i+gs_base])
    
    
            if i==0:
                plt=plot_weighted_free_energy_landscape_array(Z,plt,dim[0],dim[1],
                                              Zlabels,cmap='Spectral',contour_label=False,cbar=False,x_ticksep=60,
                                                         y_ticksep=60,x_lim=[-180,180],y_lim=[-180,180])
            elif i==1:
                plt=plot_weighted_free_energy_landscape_array(Z,plt,dim[0],dim[1],
                                              Zlabels,cmap='Spectral',contour_label=False,cbar=False,x_ticksep=60,
                                              y_ticksep=60,ylabels=False,yticks=False,x_lim=[-180,180],y_lim=[-180,180])
            else:
                plt=plot_weighted_free_energy_landscape_array(Z,plt,dim[0],dim[1],
                                              Zlabels,cmap='Spectral',contour_label=False,cbar=True,x_ticksep=60,
                                              y_ticksep=60,cbar_label="-k$_{B}$Tln(p) (kcal/mol)",ylabels=False,yticks=False,
                                                         x_lim=[-180,180],y_lim=[-180,180])  
        gs_base+=3

    
    fig.text(0.01, 0.8, 'D/E', fontsize=40, color='k')
    fig.text(0.01, 0.6, 'E/F', fontsize=40, color='k')
    fig.text(0.01, 0.40, 'F/G', fontsize=40, color='k')
    fig.text(0.01, 0.20, 'G/H', fontsize=40, color='k')

    fig.text(0.25, 0.9, '1', fontsize=40, color='k')
    fig.text(0.5, 0.9, '2', fontsize=40, color='k')
    fig.text(0.75, 0.9, '3', fontsize=40, color='k')
    
    return fig,plt


########################################################################################################################################


# Pearson Correlation Functions


def cpgl_pearsoncorr_matrix(data,data_reorder):

    Z_chain=data[:,data_reorder]
    Z_mat=np.empty((np.shape(Z_chain)[1],np.shape(Z_chain)[1]))
    for i in range(np.shape(Z_chain)[1]):
        for j in range(np.shape(Z_chain)[1]):
            Z_mat[i,j]=np.corrcoef(Z_chain[:,i],Z_chain[:,j])[0,1]

    return Z_mat

def plot_cpgl_pearsoncorr_matrix(plt,Z_mat):
    

    fig = plt.figure(figsize=(12,12)) 
    fig.subplots_adjust(hspace=0, wspace=0)
    gs = gridspec.GridSpec(1, 1)

    ax = plt.subplot(gs[0])
    im=ax.imshow(np.flipud(Z_mat),cmap="bwr",vmin=-1,vmax=1)

    cbar=ax.figure.colorbar(im)
    cbar.ax.set_ylabel('R', rotation=0, fontsize=30)
    cbar.ax.tick_params(labelsize=30)   

    corr_labels=[r"$\theta_{cp}$","$\phi_{gl}$", "$\psi_{gl}$", 
             r"$\theta_{cp}$","$\phi_{gl}$", "$\psi_{gl}$",
             r"$\theta_{cp}$","$\phi_{gl}$", "$\psi_{gl}$",
             r"$\theta_{cp}$","$\phi_{gl}$", "$\psi_{gl}$", 
             r"$\theta_{cp}$"]

    plt.yticks([y for y in range(13)],np.flipud(corr_labels),fontsize=20, rotation=0)
    plt.xticks([y for y in range(13)],corr_labels,fontsize=20, rotation=0)


    fig.text(0.01, 0.78, 'H', fontsize=30, color='k')
    fig.text(0.01, 0.64, 'G', fontsize=30, color='k')
    fig.text(0.01, 0.50, 'F', fontsize=30, color='k')
    fig.text(0.01, 0.35, 'E', fontsize=30, color='k')
    fig.text(0.01, 0.20, 'D', fontsize=30, color='k')

    fig.text(0.01, 0.72, 'G/H', fontsize=30, color='k')
    fig.text(0.01, 0.56, 'F/G', fontsize=30, color='k')
    fig.text(0.01, 0.42, 'E/F', fontsize=30, color='k')
    fig.text(0.01, 0.28, 'D/E', fontsize=30, color='k')

    fig.text(0.71, 0.12, 'H', fontsize=30, color='k')
    fig.text(0.57, 0.12, 'G', fontsize=30, color='k')
    fig.text(0.43, 0.12, 'F', fontsize=30, color='k')
    fig.text(0.29, 0.12, 'E', fontsize=30, color='k')
    fig.text(0.14, 0.12, 'D', fontsize=30, color='k')

    fig.text(0.62, 0.12, 'G/H', fontsize=30, color='k')
    fig.text(0.48, 0.12, 'F/G', fontsize=30, color='k')
    fig.text(0.34, 0.12, 'E/F', fontsize=30, color='k')
    fig.text(0.19, 0.12, 'D/E', fontsize=30, color ='k')

    return fig,plt

########################################################################################################################################

#Binary Conformational Clustering Functions for E and G rings


def binary_clusters(Z):
    """
    Assigns binary states to E and G rings for every data point/snapshot, based on partitioning regions in the theta_cp and phi_cp space of E and G rings. 
    Then creates a reduced data array with columns: data point index, E_theta_cp, E_phi_cp, G_theta_cp, G_phi_cp, binary E_state, binary G_state. 
    """
    redZ=Z[:,[2,3,6,7]]
    #E partition into 4C1 (chair) <=65 ( binary state 0), 2S0 (boat) > 65 (binary state 1)
    e=redZ[:,0]>65
    e=e.astype(int)
    #G partition into 2SO (boat) <=130 deg (binary state 0), 1C4 (chair) >130 deg (binary state 1)
    g=redZ[:,2]>130
    g=g.astype(int)

    Zcluster=np.hstack(( np.array([x for x in range(len(redZ))]).reshape(-1,1), redZ,e.reshape(-1,1),g.reshape(-1,1)))
    
    return Zcluster

def state_selection(Z):
    
    cond_len=[]
    cond_mat=[]
    Zcluster=binary_clusters(Z)
    for i in range(2):
        for j in range(2):
            cond=np.where( (Zcluster[:,5]==i)  & (Zcluster[:,6]==j) )[0] 
            cond_mat.append(cond)
            cond_len.append(len(cond))
    
    return cond_mat,cond_len,Zcluster


def convert_to_replica_frame(state_indices):

    rep = np.ceil(state_indices/200000).astype(int)
    frame = state_indices - ((rep-1)*200000)
    
    return np.transpose(np.vstack((rep,frame)))


def ring_binary_state_selection(Zcluster, ring, state_box):
    """
    Selects indices of a correpsonding ring state based on a stringent criteria within a selection box on the 2d theta/phi CP landscape. 
    state_box: Define stringent box from which to select corresponding ring state indices in terms of [[theta_cp_min, theta_cp_max], [phi_cp_min, phi_cp_max] ]
    ring: 0 for E, 1 for G
    Zcluster comes from the binary clustering of Z
    """    
    state = np.where( (Zcluster[:,2*ring+1]>state_box[0][0]) & (Zcluster[:,2*ring+1]<state_box[0][1]) & (Zcluster[:,2*ring+2]>state_box[1][0]) & (Zcluster[:,2*ring+2]<state_box[1][1]) )[0]
    
    return state


########################################################################################################################################