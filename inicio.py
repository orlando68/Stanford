# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:02:38 2019

@author: 106300
"""

from scipy.io import loadmat
import numpy as np
from scipy.linalg import block_diag
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.mplot3d import axes3d


newcolors =np.array([
                [0.6980 , 0.6980 , 0.6980 , 1],
                [0.6905 , 0.6905 , 0.6905 , 1],
                [0.6830 , 0.6830 , 0.6830 , 1],
                [0.6754 , 0.6754 , 0.6754 , 1],
                [0.6679 , 0.6679 , 0.6679 , 1],
                [0.6604 , 0.6604 , 0.6604 , 1],
                [0.6528 , 0.6528 , 0.6528 , 1],
                [0.6453 , 0.6453 , 0.6453 , 1],
                [0.6378 , 0.6378 , 0.6378 , 1],
                [0.6302 , 0.6302 , 0.6302 , 1],
                [0.6227 , 0.6227 , 0.6227 , 1],
                [0.6152 , 0.6152 , 0.6152 , 1],
                [0.6076 , 0.6076 , 0.6076 , 1],
                [0.6001 , 0.6001 , 0.6001 , 1],
                [0.5926 , 0.5926 , 0.5926 , 1],
                [0.5850 , 0.5850 , 0.5850 , 1],
                [0.5775 , 0.5775 , 0.5775 , 1],
                [0.5700 , 0.5700 , 0.5700 , 1],
                [0.5624 , 0.5624 , 0.5624 , 1],
                [0.5549 , 0.5549 , 0.5549 , 1],
                [0.5474 , 0.5474 , 0.5474 , 1],
                [0.5398 , 0.5398 , 0.5398 , 1],
                [0.5323 , 0.5323 , 0.5323 , 1],
                [0.5247 , 0.5247 , 0.5247 , 1],
                [0.5172 , 0.5172 , 0.5172 , 1],
                [0.5097 , 0.5097 , 0.5097 , 1],
                [0.5021 , 0.5021 , 0.5021 , 1],
                [0.4946 , 0.4946 , 0.4946 , 1],
                [0.4871 , 0.4871 , 0.4871 , 1],
                [0.4795 , 0.4795 , 0.4795 , 1],
                [0.4720 , 0.4720 , 0.4720 , 1],
                [0.4645 , 0.4645 , 0.4645 , 1],
                [0.4569 , 0.4569 , 0.4569 , 1],
                [0.4494 , 0.4494 , 0.4494 , 1],
                [0.4419 , 0.4419 , 0.4419 , 1],
                [0.4343 , 0.4343 , 0.4343 , 1],
                [0.4268 , 0.4268 , 0.4268 , 1],
                [0.4193 , 0.4193 , 0.4193 , 1],
                [0.4117 , 0.4117 , 0.4117 , 1],
                [0.4042 , 0.4042 , 0.4042 , 1],
                [0.3967 , 0.3967 , 0.3967 , 1],
                [0.3891 , 0.3891 , 0.3891 , 1],
                [0.3816 , 0.3816 , 0.3816 , 1],
                [0.3741 , 0.3741 , 0.3741 , 1],
                [0.3665 , 0.3665 , 0.3665 , 1],
                [0.3590 , 0.3590 , 0.3590 , 1],
                [0.3515 , 0.3515 , 0.3515 , 1],
                [0.3439 , 0.3439 , 0.3439 , 1],
                [0.3364 , 0.3364 , 0.3364 , 1],
                [0.3289 , 0.3289 , 0.3289 , 1],
                [0.3213 , 0.3213 , 0.3213 , 1],
                [0.2966 , 0.2966 , 0.2966 , 1],
                [0.2719 , 0.2719 , 0.2719 , 1],
                [0.2472 , 0.2472 , 0.2472 , 1],
                [0.2225 , 0.2225 , 0.2225 , 1],
                [0.1977 , 0.1977 , 0.1977 , 1],
                [0.1730 , 0.1730 , 0.1730 , 1],
                [0.1483 , 0.1483 , 0.1483 , 1],
                [0.1236 , 0.1236 , 0.1236 , 1],
                [0.0989 , 0.0989 , 0.0989 , 1],
                [0.0742 , 0.0742 , 0.0742 , 1],
                [0.0494 , 0.0494 , 0.0494 , 1],
                [0.0247 , 0.0247 , 0.0247 , 1], 
                [0      , 0      , 0      , 1]
                                            ])
newcmp = ListedColormap(newcolors)        
















from matplotlib import style
style.use('ggplot')

def set_numpy_decimal_places(places, width=0):
    set_np = '{0:' + str(width) + '.' + str(places) + 'f}'
    np.set_printoptions(formatter={'float': lambda x: set_np.format(x)})

def diag(elementos):
    out = elementos[0][0]
    elem = np.delete(elementos[0],0)
    for a in elem:
        out = block_diag (out,a)
    return out



#def Bar3D(vtx,color):
#
#    x3 = vtx[0,0]
#    y3 = vtx[0,1]
#    z3 = vtx[0,2]
#    dx = vtx[1,0] - vtx[0,0]
#    dy = vtx[3,1] - vtx[0,1]
#    dz = vtx[4,2] - vtx[0,2]    
#    ax.bar3d(x3, y3, z3, dx, dy, dz,color = color)

def Patch2D(vtx,color,ColMap):
    
    print()
    xy = ((vtx[0,0] + vtx[1,0])/2 , (vtx[1,1] + vtx[2,1])/2)
    d_x = abs(vtx[0,0] - vtx[1,0]) 
    d_y = abs(vtx[1,1] - vtx[2,1])
    print(xy,d_x,d_y)
#    p = plt.Rectangle(xy,d_x,d_y,color = color) 
#    plt.gca().add_patch(p)
#    plt.show()
    
    X = (vtx[0,0] + vtx[1,0])/2
    Y = (vtx[1,1] + vtx[2,1])/2
    plt.scatter(X,Y,s = d_y,c=color,cmap = ColMap,edgecolor = 'black',marker ='s',linewidth = 1, alpha = 0.75)
    plt.show()
    
def  SBAS_xPL(el,az,sigma):
    # initialize data
    HPL_prec  = 0; 
    HPL_nprec = 0;  
    VPL       = 0;
    # number of satellites
    sat_nr = int(np.size(el))
   
#    % check if there are enogh satellites
    if sat_nr > 4:
        # G matrix (RTCA do229c page J-2)
        G = np.ones((sat_nr,4))
        for i in range(sat_nr):
            
            G[i,0] = np.cos(el[i])*np.cos(az[i]);
            G[i,1] = np.cos(el[i])*np.sin(az[i]);
            G[i,2] = np.sin(el[i]);
            # G(i,4) = 1; % this one is defined with G = ones(sat_nr,4);
        # W matrix (RTCA do229c page J-2)
        W        = diag(1./sigma);
#        print('tamaño diagonal',W.shape)
        D        = np.matmul(G.T,W)
        D        = np.matmul(D,G)
        D        = linalg.inv(D)
        
        d2_east  = D[0,0]; # we do not do sqrt, since later we need it squared
        d_EN     = D[0,1];
        d2_north = D[1,1]; # we do not do sqrt, since later we need it squared
        d_U      = np.sqrt(D[2,2])
#    
        # K_H for HPL (J.2.1)
        # for non-precision approach
        K_H_nprec = 6.18;
        # for precision approach
        K_H_prec  = 6.0;
        # K_V for VPL 
        K_V       = 5.33;
        # d_major (J.1)
        d_major  = np.sqrt((d2_east+d2_north)/2 + np.sqrt(((d2_east-d2_north)/2)**2+d_EN**2));
        # HPL non-precision approach
        HPL_nprec  = K_H_nprec * d_major;
        # HPL precision approach
        HPL_prec   = K_H_prec  * d_major;
        # VPL
        VPL        = K_V       * d_U;
    return HPL_prec, HPL_nprec, VPL
#------------------------------------------------------------------------------
def hplstat(HPL,HPE,HAL,src_name):
    
    
#    p = plt.Rectangle((0,0),2,1,facecolor =[0.6980 , 0.6980 , 0.6980])
#    plt.gca().add_patch(p)
#    p = plt.Rectangle((20,20),2,1,facecolor =[0 , 0.6980 , 0.6980])
#    plt.gca().add_patch(p)
    
    newcolors =np.array([
                    [0.6980 , 0.6980 , 0.6980 , 1],
                    [0.6905 , 0.6905 , 0.6905 , 1],
                    [0.6830 , 0.6830 , 0.6830 , 1],
                    [0.6754 , 0.6754 , 0.6754 , 1],
                    [0.6679 , 0.6679 , 0.6679 , 1],
                    [0.6604 , 0.6604 , 0.6604 , 1],
                    [0.6528 , 0.6528 , 0.6528 , 1],
                    [0.6453 , 0.6453 , 0.6453 , 1],
                    [0.6378 , 0.6378 , 0.6378 , 1],
                    [0.6302 , 0.6302 , 0.6302 , 1],
                    [0.6227 , 0.6227 , 0.6227 , 1],
                    [0.6152 , 0.6152 , 0.6152 , 1],
                    [0.6076 , 0.6076 , 0.6076 , 1],
                    [0.6001 , 0.6001 , 0.6001 , 1],
                    [0.5926 , 0.5926 , 0.5926 , 1],
                    [0.5850 , 0.5850 , 0.5850 , 1],
                    [0.5775 , 0.5775 , 0.5775 , 1],
                    [0.5700 , 0.5700 , 0.5700 , 1],
                    [0.5624 , 0.5624 , 0.5624 , 1],
                    [0.5549 , 0.5549 , 0.5549 , 1],
                    [0.5474 , 0.5474 , 0.5474 , 1],
                    [0.5398 , 0.5398 , 0.5398 , 1],
                    [0.5323 , 0.5323 , 0.5323 , 1],
                    [0.5247 , 0.5247 , 0.5247 , 1],
                    [0.5172 , 0.5172 , 0.5172 , 1],
                    [0.5097 , 0.5097 , 0.5097 , 1],
                    [0.5021 , 0.5021 , 0.5021 , 1],
                    [0.4946 , 0.4946 , 0.4946 , 1],
                    [0.4871 , 0.4871 , 0.4871 , 1],
                    [0.4795 , 0.4795 , 0.4795 , 1],
                    [0.4720 , 0.4720 , 0.4720 , 1],
                    [0.4645 , 0.4645 , 0.4645 , 1],
                    [0.4569 , 0.4569 , 0.4569 , 1],
                    [0.4494 , 0.4494 , 0.4494 , 1],
                    [0.4419 , 0.4419 , 0.4419 , 1],
                    [0.4343 , 0.4343 , 0.4343 , 1],
                    [0.4268 , 0.4268 , 0.4268 , 1],
                    [0.4193 , 0.4193 , 0.4193 , 1],
                    [0.4117 , 0.4117 , 0.4117 , 1],
                    [0.4042 , 0.4042 , 0.4042 , 1],
                    [0.3967 , 0.3967 , 0.3967 , 1],
                    [0.3891 , 0.3891 , 0.3891 , 1],
                    [0.3816 , 0.3816 , 0.3816 , 1],
                    [0.3741 , 0.3741 , 0.3741 , 1],
                    [0.3665 , 0.3665 , 0.3665 , 1],
                    [0.3590 , 0.3590 , 0.3590 , 1],
                    [0.3515 , 0.3515 , 0.3515 , 1],
                    [0.3439 , 0.3439 , 0.3439 , 1],
                    [0.3364 , 0.3364 , 0.3364 , 1],
                    [0.3289 , 0.3289 , 0.3289 , 1],
                    [0.3213 , 0.3213 , 0.3213 , 1],
                    [0.2966 , 0.2966 , 0.2966 , 1],
                    [0.2719 , 0.2719 , 0.2719 , 1],
                    [0.2472 , 0.2472 , 0.2472 , 1],
                    [0.2225 , 0.2225 , 0.2225 , 1],
                    [0.1977 , 0.1977 , 0.1977 , 1],
                    [0.1730 , 0.1730 , 0.1730 , 1],
                    [0.1483 , 0.1483 , 0.1483 , 1],
                    [0.1236 , 0.1236 , 0.1236 , 1],
                    [0.0989 , 0.0989 , 0.0989 , 1],
                    [0.0742 , 0.0742 , 0.0742 , 1],
                    [0.0494 , 0.0494 , 0.0494 , 1],
                    [0.0247 , 0.0247 , 0.0247 , 1], 
                    [0      , 0      , 0      , 1]
                                                ])
    newcmp = ListedColormap(newcolors)        

    color_bar =np.array([
                 [0.6980 , 0.6980 , 0.6980],
                 [0.6905 , 0.6905 , 0.6905],
                 [0.6830 , 0.6830 , 0.6830],
                 [0.6754 , 0.6754 , 0.6754],
                 [0.6679 , 0.6679 , 0.6679],
                 [0.6604 , 0.6604 , 0.6604],
                 [0.6528 , 0.6528 , 0.6528],
                 [0.6453 , 0.6453 , 0.6453],
                 [0.6378 , 0.6378 , 0.6378],
                 [0.6302 , 0.6302 , 0.6302],
                 [0.6227 , 0.6227 , 0.6227],
                 [0.6152 , 0.6152 , 0.6152],
                 [0.6076 , 0.6076 , 0.6076],
                 [0.6001 , 0.6001 , 0.6001],
                 [0.5926 , 0.5926 , 0.5926],
                 [0.5850 , 0.5850 , 0.5850],
                 [0.5775 , 0.5775 , 0.5775],
                 [0.5700 , 0.5700 , 0.5700],
                 [0.5624 , 0.5624 , 0.5624],
                 [0.5549 , 0.5549 , 0.5549],
                 [0.5474 , 0.5474 , 0.5474],
                 [0.5398 , 0.5398 , 0.5398],
                 [0.5323 , 0.5323 , 0.5323],
                 [0.5247 , 0.5247 , 0.5247],
                 [0.5172 , 0.5172 , 0.5172],
                 [0.5097 , 0.5097 , 0.5097],
                 [0.5021 , 0.5021 , 0.5021],
                 [0.4946 , 0.4946 , 0.4946],
                 [0.4871 , 0.4871 , 0.4871],
                 [0.4795 , 0.4795 , 0.4795],
                 [0.4720 , 0.4720 , 0.4720],
                 [0.4645 , 0.4645 , 0.4645],
                 [0.4569 , 0.4569 , 0.4569],
                 [0.4494 , 0.4494 , 0.4494],
                 [0.4419 , 0.4419 , 0.4419],
                 [0.4343 , 0.4343 , 0.4343],
                 [0.4268 , 0.4268 , 0.4268],
                 [0.4193 , 0.4193 , 0.4193],
                 [0.4117 , 0.4117 , 0.4117],
                 [0.4042 , 0.4042 , 0.4042],
                 [0.3967 , 0.3967 , 0.3967],
                 [0.3891 , 0.3891 , 0.3891],
                 [0.3816 , 0.3816 , 0.3816],
                 [0.3741 , 0.3741 , 0.3741],
                 [0.3665 , 0.3665 , 0.3665],
                 [0.3590 , 0.3590 , 0.3590],
                 [0.3515 , 0.3515 , 0.3515],
                 [0.3439 , 0.3439 , 0.3439],
                 [0.3364 , 0.3364 , 0.3364],
                 [0.3289 , 0.3289 , 0.3289],
                 [0.3213 , 0.3213 , 0.3213],
                 [0.2966 , 0.2966 , 0.2966],
                 [0.2719 , 0.2719 , 0.2719],
                 [0.2472 , 0.2472 , 0.2472],
                 [0.2225 , 0.2225 , 0.2225],
                 [0.1977 , 0.1977 , 0.1977],
                 [0.1730 , 0.1730 , 0.1730],
                 [0.1483 , 0.1483 , 0.1483],
                 [0.1236 , 0.1236 , 0.1236],
                 [0.0989 , 0.0989 , 0.0989],
                 [0.0742 , 0.0742 , 0.0742],
                 [0.0494 , 0.0494 , 0.0494],
                 [0.0247 , 0.0247 , 0.0247], 
                 [0 , 0 , 0]
                                             ])
    


    
    # colormap(color_bar)
    
    # other colors used in the plot
    #color1 = [134/255 134/255 134/255] % - dark gray
    color1 = (1,0.1,0.1)              # - red (original) 
    
    #color2 = [174/255 174/255 174/255] % - middle gray
    color2 = (1,0.55,0.55)            # - pink (original)
    
    #color3 = [222/255 222/255 222/255] % - light gray
    color3 = (1,1,0.5)                 #- yellow (original)
    #
    #color4 = color2                    % - middle gray
    color4 = (1,.55,0.3)         

    # size of VPL, which should be the same for VPE as well
    n = HPL.shape[1]

    # HPE
    j0 = np.floor(2.0*np.abs(HPE))#+1
    j0[j0>99] = 99;
    j0 = j0.astype(int)
    
    # HPL
    HPL = HPL[0]
    k0 = np.floor(2.0*abs(HPL))#+1
    k0[k0>99] = 99;
    k0 = k0.astype(int)
#    print(k)
#    print('tamaño de k',k.shape)
#    print('tamaño de j',j.shape)
   
    
    # initialize hpl_stat
    data     = np.zeros((100,100))
    diagonal = np.zeros((100,1))
    for i in range(n):
        # statistics
#        print(i,k[i],j[i])
        data[k0[i],j0[i]] = data[k0[i],j0[i]]+1;
        # diagonal
        bool1 = (k0[i] == j0[i])
        bool2 = (abs(HPE[i]) < abs(HPL[i]))
#        print(bool2.all())
        if bool1.all() and bool2.all() :
            diagonal[k0[i],1] = diagonal[k0[i],1]+1;
        
    c = 1
    colors = 'brygcmw';
#    clf
    
    err_bin  = np.arange(0.25,49.75+0.5,0.5)
    sig_bin  = np.arange(0.25,49.75+0.5,0.5)
    
    seconds  = n;
    sec_available = n;
    diag_cnt = np.sum(diagonal);
    
    # determine the number of points and axis ranges
    n_pts = np.sum(np.sum(data,axis= 0));
    if sec_available == 1:
        epochs = seconds
    else:
        epochs = n_pts

    
    
    d_err_bin = np.mean(np.diff(err_bin));
    x_lo_bnd  = np.min(err_bin) - d_err_bin/2;
    x_up_bnd  = np.max(err_bin) + d_err_bin/2;
    
    d_sig_bin = np.mean(np.diff(sig_bin));
    y_lo_bnd  = np.min(sig_bin) - d_sig_bin/2;
    y_up_bnd  = np.max(sig_bin) + d_sig_bin/2;
    
    z_lo_bnd  = 1;
    z_up_bnd  = np.max(data);
    
    # clear plot
#    clf
    
    # plot each data point as a pixel

    i0,j0  = np.where(data>0) 
    i0 = i0.astype(int)
    j0 = j0.astype(int)

    base1 = np.array([-0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5])
    base2 = np.array([-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5])

    w = 15
    h = 10
    d = 70
    plt.figure(figsize=(w, h), dpi=d)
    for idx in range(np.size(i0)):
#        print(np.size(i0),idx)
        z = np.log10(data[i0[idx],j0[idx]])
        col1 = err_bin[j0[idx]] + base1*d_err_bin
#        print('<<<<<<<',np.size(sig_bin),i0[idx])
        col2 = sig_bin[i0[idx]] + base2*d_sig_bin
        vtx_mat = np.array([col1,col2,[0,0,0,0,z,z,z,z]])
        vtx_mat = vtx_mat.T
        c_idx = int(np.ceil(63*(np.log10(data[i0[idx],j0[idx]])/np.log10(z_up_bnd))))# + 1
        print(c_idx)
#        print(color_bar[c_idx,:])
#        Bar3D( vtx_mat ,color_bar[c_idx,:])
#        Patch2D( vtx_mat ,color_bar[c_idx,:])
        X = (vtx_mat[0,0] + vtx_mat[1,0])/2
        Y = (vtx_mat[1,1] + vtx_mat[2,1])/2
        
        plt.scatter(X,Y,s = 1,c=c_idx,cmap = newcmp,marker ='s', alpha = 1)
        p = plt.Rectangle((X,Y),0.5,0.5,facecolor =newcolors[c_idx,:])
        plt.gca().add_patch(p)
        
    cbar = plt.colorbar()
    cbar.set_label('amplitude')
#    plt.colorbar(color_bar)
    plt.axis([0,50,0,50])
    plt.show()

#------------------------------------------------------------------------------

    
#------------------------------------------------------------------------------
annots               = loadmat('easy14_data.mat')
az_EGNOS             = annots['az_EGNOS']
el_EGNOS             = annots['el_EGNOS']
end_x                = annots['end_x']
mask_EGNOS           = annots['mask_EGNOS']
rec_pos_utm_egnos    = annots['rec_pos_utm_egnos']
rec_pos_utm_gps      = annots['rec_pos_utm_gps']
recpos_LatLong_EGNOS = annots['recpos_LatLong_EGNOS']
recpos_LatLong_GPS   = annots['recpos_LatLong_GPS']
recpos_XYZ_EGNOS     = annots['recpos_XYZ_EGNOS']
recpos_XYZ_GPS       = annots['recpos_XYZ_GPS']
ref_pos_UTM          = annots['ref_pos_UTM']
ref_pos_XYZ          = annots['ref_pos_XYZ']
sigma_EGNOS_air      = annots['sigma_EGNOS_air']
sigma_EGNOS_flt      = annots['sigma_EGNOS_flt']
sigma_EGNOS_iono     = annots['sigma_EGNOS_iono']
sigma_EGNOS_tropo    = annots['sigma_EGNOS_tropo']
start_x              = annots['start_x']

# Horizontal alert limit
HAL = 35;
# Vertical alert limits
VAL1 = 12;
VAL2 = 20;


# epoch count
num_epoch   = recpos_XYZ_GPS.shape[0]
# epoch count in hours
value       = num_epoch/3600
epoch_count = '%.2f hours' % value

# initialization
rec_pos_UTM = np.zeros((num_epoch,6))
HPL         = np.zeros((num_epoch,1))
VPL         = np.zeros((num_epoch,1))
sat_nr      = np.zeros((num_epoch,1))

# get coordinate errors for GPS only
rec_pos_UTM[:,0] = rec_pos_utm_gps[:,0]-ref_pos_UTM[0][0];
rec_pos_UTM[:,1] = rec_pos_utm_gps[:,1]-ref_pos_UTM[0][1];
rec_pos_UTM[:,2] = rec_pos_utm_gps[:,2]-ref_pos_UTM[0][2];
# for EGNOS corrected coordinates
rec_pos_UTM[:,3] = rec_pos_utm_egnos[:,0]-ref_pos_UTM[0][0];
rec_pos_UTM[:,4] = rec_pos_utm_egnos[:,1]-ref_pos_UTM[0][1];
rec_pos_UTM[:,5] = rec_pos_utm_egnos[:,2]-ref_pos_UTM[0][2];

# compute HPE for EGNOS data
HPE = (rec_pos_UTM[:,3]**2 + rec_pos_UTM[:,4]**2)**0.5;
# get VPE for EGNOS data
VPE = rec_pos_UTM[:,5];

for i in range(num_epoch):
    index = np.nonzero(mask_EGNOS[i,:])
    sat_nr[i,0] = np.size(index);
    set_numpy_decimal_places(4, 6)
    sigmas_all = sigma_EGNOS_flt[i,index] + sigma_EGNOS_iono[i,index] + sigma_EGNOS_tropo[i,index] + sigma_EGNOS_air[i,index];

    # compute HPL and VPL
#    print(np.size(el_EGNOS[i,index]))
#    HPL(i,0), kk , VPL(i,0) = SBAS_xPL(el_EGNOS(i,index),az_EGNOS(i,index),sigmas_all);
    HPL[i,0], kk , VPL[i,0] = SBAS_xPL(el_EGNOS[i,index][0],az_EGNOS[i,index][0],sigmas_all);
"""
#------------------------------------------------------------------------------    
plt.figure(1)
plt.plot(rec_pos_UTM[:,0] , rec_pos_UTM[:,1],'k.')
plt.plot(rec_pos_UTM[:,3],rec_pos_UTM[:,4],'r.')
plt.show()
#------------------------------------------------------------------------------ 
plt.figure(2)
plt.subplot(3,1,1)
plt.plot(range(rec_pos_UTM.shape[0]),rec_pos_UTM[:,0],'-.b')
plt.plot(range(rec_pos_UTM.shape[0]),rec_pos_UTM[:,3],'-r')
plt.subplot(3,1,2)
plt.plot(range(rec_pos_UTM.shape[0]),rec_pos_UTM[:,1],'-.b')
plt.plot(range(rec_pos_UTM.shape[0]),rec_pos_UTM[:,4],'-r')
plt.subplot(3,1,3)
plt.plot(range(rec_pos_UTM.shape[0]),rec_pos_UTM[:,2],'-.b')
plt.plot(range(rec_pos_UTM.shape[0]),rec_pos_UTM[:,5],'-r')
"""
#  Plot 4: Vertical Stanford plot 
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d', facecolor='w') 

hplstat(HPL.T,HPE,HAL,'EGNOS')

#ax.set_xlim3d(0, 50)
#ax.set_zlim3d(0, 10)
#ax.view_init(90, 270)
#plt.show()
