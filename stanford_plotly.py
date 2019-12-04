# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:02:38 2019

@author: 106300
"""

from scipy.io import loadmat
import numpy as np
from scipy.linalg import block_diag
from scipy import linalg
import matplotlib as mpl
import pickle
import matplotlib.pyplot as plt
#from matplotlib.patches import Rectangle
#from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
#from mpl_toolkits.mplot3d import axes3d
import matplotlib.patches as patches
import pandas as pd
from matplotlib import cm
from collections import OrderedDict
import plotly.graph_objects as go

from plotly.subplots import make_subplots


#------------------------------------------------------------------------------
def set_numpy_decimal_places(places, width=0):
    set_np = '{0:' + str(width) + '.' + str(places) + 'f}'
    np.set_printoptions(formatter={'float': lambda x: set_np.format(x)})
#------------------------------------------------------------------------------
#def diag(elementos):
#    out = elementos[0][0]
#    elem = np.delete(elementos[0],0)
#    for a in elem:
#        out = block_diag (out,a)
#    return out
#------------------------------------------------------------------------------

def Reverse_ColorBar(Bar_input):
    Bar_output = np.zeros(Bar_input.shape)
    for k in range(Bar_input.shape[0]):
        Bar_output[k,:] = Bar_input[Bar_input.shape[0]-1-k,:] 
    return Bar_output

#color_bar =np.array([
#[0.6980 , 0.6980 , 0.6980 , 1],[0.6905 , 0.6905 , 0.6905 , 1],[0.6830 , 0.6830 , 0.6830 , 1],[0.6754 , 0.6754 , 0.6754 , 1],
#[0.6679 , 0.6679 , 0.6679 , 1],[0.6604 , 0.6604 , 0.6604 , 1],[0.6528 , 0.6528 , 0.6528 , 1],[0.6453 , 0.6453 , 0.6453 , 1],
#[0.6378 , 0.6378 , 0.6378 , 1],[0.6302 , 0.6302 , 0.6302 , 1],[0.6227 , 0.6227 , 0.6227 , 1],[0.6152 , 0.6152 , 0.6152 , 1],
#[0.6076 , 0.6076 , 0.6076 , 1],[0.6001 , 0.6001 , 0.6001 , 1],[0.5926 , 0.5926 , 0.5926 , 1],[0.5850 , 0.5850 , 0.5850 , 1],
#[0.5775 , 0.5775 , 0.5775 , 1],[0.5700 , 0.5700 , 0.5700 , 1],[0.5624 , 0.5624 , 0.5624 , 1],[0.5549 , 0.5549 , 0.5549 , 1],
#[0.5474 , 0.5474 , 0.5474 , 1],[0.5398 , 0.5398 , 0.5398 , 1],[0.5323 , 0.5323 , 0.5323 , 1],[0.5247 , 0.5247 , 0.5247 , 1],
#[0.5172 , 0.5172 , 0.5172 , 1],[0.5097 , 0.5097 , 0.5097 , 1],[0.5021 , 0.5021 , 0.5021 , 1],[0.4946 , 0.4946 , 0.4946 , 1],
#[0.4871 , 0.4871 , 0.4871 , 1],[0.4795 , 0.4795 , 0.4795 , 1],[0.4720 , 0.4720 , 0.4720 , 1],[0.4645 , 0.4645 , 0.4645 , 1],
#[0.4569 , 0.4569 , 0.4569 , 1],[0.4494 , 0.4494 , 0.4494 , 1],[0.4419 , 0.4419 , 0.4419 , 1],[0.4343 , 0.4343 , 0.4343 , 1],
#[0.4268 , 0.4268 , 0.4268 , 1],[0.4193 , 0.4193 , 0.4193 , 1],[0.4117 , 0.4117 , 0.4117 , 1],[0.4042 , 0.4042 , 0.4042 , 1],
#[0.3967 , 0.3967 , 0.3967 , 1],[0.3891 , 0.3891 , 0.3891 , 1],[0.3816 , 0.3816 , 0.3816 , 1],[0.3741 , 0.3741 , 0.3741 , 1],
#[0.3665 , 0.3665 , 0.3665 , 1],[0.3590 , 0.3590 , 0.3590 , 1],[0.3515 , 0.3515 , 0.3515 , 1],[0.3439 , 0.3439 , 0.3439 , 1],
#[0.3364 , 0.3364 , 0.3364 , 1],[0.3289 , 0.3289 , 0.3289 , 1],[0.3213 , 0.3213 , 0.3213 , 1],[0.2966 , 0.2966 , 0.2966 , 1],
#[0.2719 , 0.2719 , 0.2719 , 1],[0.2472 , 0.2472 , 0.2472 , 1],[0.2225 , 0.2225 , 0.2225 , 1],[0.1977 , 0.1977 , 0.1977 , 1],
#[0.1730 , 0.1730 , 0.1730 , 1],[0.1483 , 0.1483 , 0.1483 , 1],[0.1236 , 0.1236 , 0.1236 , 1],[0.0989 , 0.0989 , 0.0989 , 1],
#[0.0742 , 0.0742 , 0.0742 , 1],[0.0494 , 0.0494 , 0.0494 , 1],[0.0247 , 0.0247 , 0.0247 , 1],[0      , 0      , 0      , 1]     ])
#matplt_cmap = ListedColormap(color_bar)

def ColorMap():
    cmp_list    = [ 'viridis','Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                    'YlOrBr' , 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    c_map        = cm.get_cmap(cmp_list[0]+'', 256) # ''= normal, '_r' = reversed
    color_BAR   = c_map(np.linspace(0, 1, 64))
    
    matlab      = 'inicial'    #-----------el de Matlab---------------
    matlab      = 'new'
    if matlab == 'inicial': 
        color_BAR_r = Reverse_ColorBar(color_BAR)
        c_map = ListedColormap(color_BAR_r)  
    return c_map,color_BAR
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def barra_RGB(color_BAR):
#    'rgb(0.9553  , 0.901065, 0.118128)'
    color_BAR_RGB = []
    for color in color_BAR:
        color_RGB = 'rgb('+str(color[0])+','+str(color[1])+','+str(color[2])+')'
        color_BAR_RGB.append(color_RGB)
        
    return color_BAR_RGB
    
#------------------------------------------------------------------------------
def plot_ColorBar(dib,RGB_bar,Vref):
    D    = Vref/64
    d    = D/2
    
    for k in range(64):
        p_x = 1
        p_y = k*D+d
        dib.add_shape(go.layout.Shape(type="rect",x0=p_x-d,y0=p_y-d,x1=p_x+d,y1=p_y+d,
                                      line=dict(color=RGB_bar[k]),
                                      fillcolor=RGB_bar[k]),secondary_y=True,row=1, col=2)
    dib.update_xaxes(range=[0.5, 1], showgrid=False,title_text="",showticklabels=False, row=1, col=2)
#    dib.update_yaxes(range=[0, Vref], showgrid=False ,title_text="yaxis izda2 title" ,secondary_y=False, row=1, col=2)
    dib.update_yaxes(range=[0, Vref], showgrid=False ,title_text='Number of Points per Pixel' ,secondary_y=True, row=1, col=2)
    
    return
#------------------------------------------------------------------------------
def add_text(dib,texto,x0,y0,color):
    dib.add_trace(go.Scatter(x=x0,y=y0,text=texto, mode="text",orientation ='v',textposition="middle center",
#                             textfont=dict(family="sans serif", size=18, color=color )
                             ), row=1, col=1)
    return
#------------------------------------------------------------------------------
def add_patch(dib,p_x,p_y, color):
    dib.add_shape(go.layout.Shape(type="rect",x0=p_x-0.25,y0=p_y-0.25,x1=p_x+0.25,y1=p_y+0.25,
                              line=dict(color=color), fillcolor=color),layer = "below", row=1, col=1 )
    return
#------------------------------------------------------------------------------
def add_polygon(dib,x_points,y_points,color):
    camino = 'M' 
    for index,k in enumerate(x_points):
        camino = camino + ' '+ str(x_points[index]) +','+str(y_points[index])
        if index < np.size(x_points)-1:
            camino = camino + ' L'
    camino = camino + ' Z'
    print(camino)

    dib.add_shape(go.layout.Shape(type="path",path=camino,fillcolor=color),line_color=color,layer = "below", opacity=1.0,row=1, col=1)
    
    return
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def bound2(percent,*argv):
    if len(argv) == 1:
        data  = argv[0]
        total = np.sum(data,axis=0) # 1xN array
    if len(argv) == 2:
        data  = argv[0]
        total = argv[1]
        
    data    = np.matrix(data)  
    m,n     = data.shape # m-rows n-columns
    if m == 1 :
        data = data.T
        m,n     = data.shape    
    index   = np.zeros(n)
    bnd_cnt = np.ceil(percent*total)
    if np.size(bnd_cnt) == 1:
        bnd_cnt = np.array([bnd_cnt])
    for idx in range(n):# 1:n
        include = 0
        start   = 0  #1;
#        print('start0',start,'idx=',idx)
        curr    = data[start,idx]
        while (curr < bnd_cnt[idx]) and (include < m-start):
            include = include + 1
#            print('include=',include)
            curr    = curr + data[start+include,idx]
        index[idx] = start + include
#        print('index=>>>',index)
    index= index.astype('int')
    return index
#------------------------------------------------------------------------------
  
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
#        W        = diag(1/sigma);
        W        = np.diag(1/sigma[0]);
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
#                                VPLSTAT
#------------------------------------------------------------------------------
def vplstat(VPL,VPE,VAL1,VAL2,src_name):
    # other colors used in the plot
    #color1 = [134/255 134/255 134/255] % - dark gray
    color1 = (1,0.1,0.1,0.75)              # - red (original) 
    #color2 = [174/255 174/255 174/255] % - middle gray
    color2 = (1,0.55,0.55,0.75)            # - pink (original)
    #color3 = [222/255 222/255 222/255] % - light gray
    color3 = (1,1,0.5,0.75)                 #- yellow (original)
    #color4 = color2                    % - middle gray
    color4 = (1,.55,0.30,.75)    
     
    # size of VPL, which should be the same for VPE as well
    n          = VPL.shape[1]
    VPL        = VPL[0]
    # VPE
    j0         = np.floor(4.0*np.abs(VPE)) #+1
    j0[j0>=99] = 99
    j0         = j0.astype(int)
    # VPL
    k0         = np.floor(4.0*np.abs(VPL)) #+1
    k0[k0>=99] = 99;
    k0         = k0.astype(int)
#    k0         = k0[0]
    # initialize vpl_stat
    data       = np.zeros((100,100))
    diagonal   = np.zeros((100,1))
    for i in range(n):
        # statistics
        data[k0[i],j0[i]] = data[k0[i],j0[i]]+1
        # diagonal
        bool1 = (k0[i] == j0[i])
        bool2 = abs(VPE[i]) < abs(VPL[i])
        if bool1.all() and bool2.all():
            diagonal[k0[i],1] = diagonal[k0[i],1]+1

    
    err_bin       = np.arange(0.125,24.875+0.25,0.25)
    sig_bin       = np.arange(0.125,24.875+0.25,0.25)

    seconds       = n
    sec_available = n
    diag_cnt      = np.sum(diagonal);

    # determine the number of points and axis ranges
    n_pts = np.sum(np.sum(data,axis = 0))
    if sec_available == 1:
        epochs = seconds
    else:
        epochs = n_pts

    d_err_bin = np.mean(np.diff(err_bin))
    x_lo_bnd  = np.min(err_bin) - d_err_bin/2
    x_up_bnd  = np.max(err_bin) + d_err_bin/2
    
    d_sig_bin = np.mean(np.diff(sig_bin))
    y_lo_bnd  = np.min(sig_bin) - d_sig_bin/2
    y_up_bnd  = np.max(sig_bin) + d_sig_bin/2
    
    z_lo_bnd  = 1
    z_up_bnd  = np.max(data)
   
    # plot each data point as a pixel
    j0,i0     = np.where(data.T>0) 
    i0        = i0.astype(int)
    j0        = j0.astype(int)
    
    base1     = np.array([-0.5, 0.5 , 0.5, -0.5, -0.5, 0.5 , 0.5, -0.5])
    base2     = np.array([-0.5, -0.5, 0.5, 0.5 , -0.5, -0.5, 0.5, 0.5 ])    
    df_STFD = pd.DataFrame(columns = ['X','Y','Z','c_idx'])
    for idx in range(np.size(i0)):
        z                        = np.log10(data[i0[idx],j0[idx]])
        col1                     = err_bin[j0[idx]] + base1*d_err_bin
        col2                     = sig_bin[i0[idx]] + base2*d_sig_bin
        vtx_mat                  = np.array([col1,col2,[0,0,0,0,z,z,z,z]])
        vtx_mat                  = vtx_mat.T
#        print(vtx_mat)
#        print(np.ceil(63*(np.log10(data[i0[idx],j0[idx]])/np.log10(z_up_bnd))))
        c_idx                    = int(np.ceil(63*(np.log10(data[i0[idx],j0[idx]])/np.log10(z_up_bnd))))# + 1
        X                        = (vtx_mat[0,0] + vtx_mat[1,0])/2
        Y                        = (vtx_mat[1,1] + vtx_mat[2,1])/2
        Z                        = z
        df_STFD.loc[idx,'X']     = X
        df_STFD.loc[idx,'Y']     = Y
        df_STFD.loc[idx,'Z']     = Z
        df_STFD.loc[idx,'c_idx'] = c_idx

#==============================================================================
    plt.figure(figsize=(12,10), dpi=70)
    matplt_cmap,color_bar = ColorMap()
    norm        = mpl.colors.Normalize(vmin=z_lo_bnd, vmax=z_up_bnd)   #-----aqui normalizamos
    
    ax1         = plt.axes([0.1-0.014, 0.1, .75, .8])

    for k in df_STFD.index:
        rect = patches.Rectangle( (df_STFD.loc[k,'X']-0.25, df_STFD.loc[k,'Y']-0.25),
                                 0.5, 0.5, facecolor = color_bar[df_STFD.loc[k,'c_idx']])
        ax1.add_patch(rect)    
    ax1.axis([x_lo_bnd, x_up_bnd,y_lo_bnd, y_up_bnd])
    ax1.set_xlabel('Error [m]')
    src_name= 'EGNOS'
    ax1.set_ylabel(r'$HPL_{'+src_name+'} [m]$')
    ax1.set_title('Horizontal Performance ['+str(seconds)+' seconds]')
    ax1.set_aspect('equal', 'box') 
    
    ax2         = plt.axes([.87-0.014, .1, .025, .8])
    cb1         = mpl.colorbar.ColorbarBase(ax2, cmap = matplt_cmap, norm = norm, orientation = 'vertical')    
    cb1.set_label('Number of Points per Pixel')


    # determine availability and # of integrity failures

    i_diag1 = np.unique(np.append( np.where(err_bin == VAL1)[0],
                                   np.where(err_bin == VAL2)[0] ) )
    
    i_diag2 = np.where(err_bin < VAL1)[0]
    i_diag3 = np.where(err_bin > VAL2)[0]

    cond1   = np.where(err_bin > VAL1)[0]
    cond2   = np.where(err_bin < VAL2)[0]
    i_diag4 = np.intersect1d(cond1, cond2)

    cond1   = np.intersect1d(np.where(err_bin[j0] >= VAL1)[0], np.where(sig_bin[i0] < VAL1)[0])
    cond2   = np.intersect1d(np.where(err_bin[j0] >= VAL2)[0], np.where(sig_bin[i0] < VAL2)[0])
    i_fail1 = np.unique(np.append( cond1,cond2 ))
    n_fail1 = np.sum(np.sum(np.diag(data[i0[i_fail1],j0[i_fail1]]))) - np.sum(diagonal[i_diag1])
    
    i_fail2 = np.intersect1d(   np.where(err_bin[j0]/sig_bin[i0] >=1.0)[0] , np.where(err_bin[j0] < VAL1)[0] )
    n_fail2 = np.sum(np.sum(np.diag(data[i0[i_fail2],j0[i_fail2]])))- np.sum(diagonal[i_diag2])
    
    i_fail3 = np.intersect1d(   np.where(err_bin[j0]/sig_bin[i0] >=1.0)[0] , np.where(sig_bin[j0] < VAL2)[0] )
    n_fail3 = np.sum(np.sum(np.diag(data[i0[i_fail3],j0[i_fail3]]))) - np.sum(diagonal[i_diag3])
    
    i_fail4 = np.intersect1d(   np.where(err_bin[j0]/sig_bin[i0] >=1.0)[0] , np.where(sig_bin[i0] < VAL1)[0] )
    i_fail4 = np.intersect1d(  i_fail4 , np.where(err_bin[j0] < VAL2)[0] ) 
    n_fail4 = np.sum(np.sum(np.diag(data[i0[i_fail4],j0[i_fail4]]))) - np.sum(diagonal[i_diag4])
    
    i_cont  = np.where(sig_bin[i0] >= VAL2);
    n_cont  = np.sum(np.sum(np.diag(data[i0[i_cont],j0[i_cont]])));
    
    i_avail1 = np.intersect1d( np.where(err_bin[j0]/sig_bin[i0] < 1.0)[0],  np.where (sig_bin[i0] < VAL1)[0])
    n_avail1 = np.sum(np.sum(np.diag(data[i0[i_avail1],j0[i_avail1]]))) + np.sum(diagonal[i_diag2]);

    i_avail2 = np.intersect1d( np.where(err_bin[j0]/sig_bin[i0] < 1.0)[0], np.where(sig_bin[i0] < VAL2)[0])
    n_avail2 = np.sum(np.sum(np.diag(data[i0[i_avail2],j0[i_avail2]]))) + np.sum(diagonal[np.append(i_diag2, i_diag4)])


    
#     show the region of IPV operation
    HT = ax1.text(0.57*(VAL2 - x_lo_bnd) + x_lo_bnd, 0.95*(VAL2 - y_lo_bnd) + y_lo_bnd,'IPV Operation',fontsize=10,ha='center');

    if n_avail2/epochs >= 0.999995:
        HT = ax1.text(0.53*(VAL2 - x_lo_bnd) + x_lo_bnd,0.89*(VAL2 - y_lo_bnd) + y_lo_bnd,'> 99.999%');

    else:
        HT = ax1.text(0.57*(VAL2 - x_lo_bnd) + x_lo_bnd,0.89*(VAL2 - y_lo_bnd) + y_lo_bnd, "{:.3f}".format(100.0*n_avail2/epochs)+'%')
  
    # show the region of CAT I operation
    HT = ax1.text(0.45*(VAL1 - x_lo_bnd) + x_lo_bnd,0.93*(VAL1 - y_lo_bnd) + y_lo_bnd,'CAT I Oper.',fontsize=10)

    ax1.text(0.45*(VAL1 - x_lo_bnd) + x_lo_bnd,0.93*(VAL1 - y_lo_bnd) + y_lo_bnd,'CAT I Oper.',fontsize=10)
    if n_avail1/epochs >= 0.999995:
      HT = ax1.text(0.4*(VAL1 - x_lo_bnd) + x_lo_bnd,0.84*(VAL1 - y_lo_bnd) + y_lo_bnd,'> 99.999%');

    else:
        HT = ax1.text(0.45*(VAL1 - x_lo_bnd) + x_lo_bnd,0.84*(VAL1 - y_lo_bnd) + y_lo_bnd,"{:.3f}".format(100.0*n_avail1/epochs)+'%')

    # outline the region of integrity failures
    coor = np.array([[VAL1, VAL1, VAL2, VAL2, x_up_bnd, x_up_bnd],[y_lo_bnd, VAL1, VAL1, VAL2, VAL2, y_lo_bnd]])
    ax1.add_patch(patches.Polygon(coor.T, linewidth=10,facecolor=color1,zorder=0))
    HT   = ax1.text(VAL2, 0.5*(VAL2 - y_lo_bnd) + y_lo_bnd,'HMI');
    HT   = ax1.text(VAL2, 0.4*(VAL2 - y_lo_bnd) + y_lo_bnd,'epochs: '+ str(n_fail1))

    # outline the lowest region of VPL failures
    coor = np.array([[x_lo_bnd, VAL1, VAL1],[y_lo_bnd, VAL1, y_lo_bnd]])
    ax1.add_patch(patches.Polygon(coor.T, linewidth=10,facecolor=color2,zorder=0))
    HT   = ax1.text(0.67*(VAL1 - x_lo_bnd) + x_lo_bnd, 0.35*(VAL1 - y_lo_bnd) + y_lo_bnd, 'MI',ha= 'center')
    HT   = ax1.text(0.67*(VAL1 - x_lo_bnd) + x_lo_bnd,0.25*(VAL1 - y_lo_bnd) + y_lo_bnd,'epochs: '+ str(n_fail2),ha= 'center')

    # outline the middle region of VPL failures

    coor = np.array([[VAL1, VAL2, VAL2],  [VAL1, VAL2, VAL1]])
    ax1.add_patch(patches.Polygon(coor.T, linewidth=10,facecolor=color2,zorder=0))
    HT   = ax1.text(0.67*(VAL2 - VAL1) + VAL1,0.39*(VAL2 - VAL1) + VAL1, 'MI',ha= 'center')
    HT   = ax1.text(0.67*(VAL2 - VAL1) + VAL1,0.25*(VAL2 - VAL1) + VAL1, 'epochs: '+ str(n_fail4),ha= 'center')

    # outline the region of unavailability
    coor = np.array([[x_lo_bnd, x_up_bnd, x_up_bnd, x_lo_bnd],[VAL2, VAL2, y_up_bnd, y_up_bnd]  ])
    ax1.add_patch(patches.Polygon(coor.T, linewidth=10,facecolor=color3,zorder=0))
    HT   = ax1.text(0.50*(x_up_bnd - x_lo_bnd) + x_lo_bnd,0.65*(y_up_bnd - VAL2) + VAL2,'System Unavailable',fontsize=10,ha= 'center')
    HT   = ax1.text(0.50*(x_up_bnd - x_lo_bnd) + x_lo_bnd,0.35*(y_up_bnd - VAL2) + VAL2,'Alarm Epochs: '+ str(n_cont),ha= 'center')

#     outline the region where integrity failures and unavailability overlap
    coor = np.array([[VAL2, x_up_bnd, x_up_bnd],[VAL2, y_up_bnd, VAL2] ])
    ax1.add_patch(patches.Polygon(coor.T, linewidth=10,facecolor=color4,zorder=0))
    HT   = ax1.text(0.70*(x_up_bnd - VAL2) + VAL2, 0.4*(y_up_bnd - VAL2) + VAL2, 'MI:',ha= 'center')
    HT   = ax1.text(0.70*(x_up_bnd - VAL2) + VAL2, 0.175*(y_up_bnd - VAL2) + VAL2,str(n_fail3))

#% label the axes and add a title
    ax1.set_xlabel('Error [m]',fontsize='xx-large')
    ax1.set_ylabel(r'$VPL_{'+ src_name +'} [m]$',fontsize='xx-large');
    ax1.set_title ('Vertical Performance [' +str(seconds)+' seconds]',fontsize='xx-large')

#    bnd_68 =err_bin[bound2(0.68 ,np.sum(data,axis=0).T)] + d_err_bin/2;
#    bnd_95 =err_bin[bound2(0.95 ,np.sum(data,axis=0).T)] + d_err_bin/2;
#    bnd_999=err_bin[bound2(0.999,np.sum(data,axis=0).T)] + d_err_bin/2;
    
    err_bnd_68  = err_bin[bound2(0.68 , np.sum(data,axis=0).T)] + d_err_bin/2;
    err_bnd_95  = err_bin[bound2(0.95 , np.sum(data,axis=0).T)] + d_err_bin/2;
    err_bnd_999 = err_bin[bound2(0.999, np.sum(data,axis=0).T)] + d_err_bin/2;

    coor        = np.array([[err_bnd_68 + 0.01*(x_up_bnd - x_lo_bnd), err_bnd_68, err_bnd_68 - 0.01*(x_up_bnd - x_lo_bnd)],[y_lo_bnd, 0.02*(y_up_bnd - y_lo_bnd), y_lo_bnd]  ])
    ax1.add_patch(patches.Polygon(coor.T, linewidth=10,facecolor='k',zorder=0))
    HT          =ax1.text(err_bnd_68, 0.03*(y_up_bnd - y_lo_bnd), '68%', rotation=90, verticalalignment =  'bottom',fontsize=10);

    coor        = np.array([[err_bnd_95 + 0.01*(x_up_bnd - x_lo_bnd), err_bnd_95, err_bnd_95 - 0.01*(x_up_bnd - x_lo_bnd)],[y_lo_bnd, 0.02*(y_up_bnd - y_lo_bnd), y_lo_bnd] ])
    ax1.add_patch(patches.Polygon(coor.T, linewidth=10,facecolor='k',zorder=0))
    HT= ax1.text(err_bnd_95, 0.03*(y_up_bnd - y_lo_bnd), '95%', rotation=90, verticalalignment =  'bottom',fontsize=10);

    coor        = np.array([[err_bnd_999 + 0.01*(x_up_bnd - x_lo_bnd), err_bnd_999, err_bnd_999 - 0.01*(x_up_bnd - x_lo_bnd)], [y_lo_bnd, 0.02*(y_up_bnd - y_lo_bnd), y_lo_bnd] ])
    ax1.add_patch(patches.Polygon(coor.T, linewidth=10,facecolor='k',zorder=0))
    HT          =ax1.text(err_bnd_999, 0.03*(y_up_bnd - y_lo_bnd),'99.9%', rotation=90, verticalalignment =  'bottom',fontsize=10);

    sig_bnd_68  =sig_bin[bound2(0.68 ,np.sum(data.T,axis=0).T, epochs)] + d_sig_bin/2
    sig_bnd_95  =sig_bin[bound2(0.95 ,np.sum(data.T,axis=0).T, epochs)] + d_sig_bin/2
    sig_bnd_999 =sig_bin[bound2(0.999,np.sum(data.T,axis=0).T, epochs)] + d_sig_bin/2

    coor        = np.array([[x_up_bnd, 0.98*(x_up_bnd - x_lo_bnd) + x_lo_bnd, x_up_bnd],[sig_bnd_68 + 0.01*(y_up_bnd - y_lo_bnd), sig_bnd_68, sig_bnd_68 - 0.01*(y_up_bnd - y_lo_bnd)] ])
    ax1.add_patch(patches.Polygon(coor.T, linewidth=10,facecolor='k',zorder=0))

    ax1.text(0.925*(x_up_bnd - x_lo_bnd) + x_lo_bnd, sig_bnd_68, '68%',fontsize=10);
    coor        = np.array([[x_up_bnd, 0.98*(x_up_bnd - x_lo_bnd) + x_lo_bnd, x_up_bnd],[sig_bnd_95 + 0.01*(y_up_bnd - y_lo_bnd), sig_bnd_95, sig_bnd_95 - 0.01*(y_up_bnd - y_lo_bnd)]])
    ax1.add_patch(patches.Polygon(coor.T, linewidth=10,facecolor='k',zorder=0))
  
    ax1.text(0.925*(x_up_bnd - x_lo_bnd) + x_lo_bnd, sig_bnd_95, '95%',fontsize=10);
    coor        = np.array([[x_up_bnd, 0.98*(x_up_bnd - x_lo_bnd) + x_lo_bnd, x_up_bnd], [sig_bnd_999 + 0.01*(y_up_bnd - y_lo_bnd), sig_bnd_999, sig_bnd_999 - 0.01*(y_up_bnd - y_lo_bnd)]])
    ax1.add_patch(patches.Polygon(coor.T, linewidth=10,facecolor='k',zorder=0))
   
    ax1.text(0.9*(x_up_bnd - x_lo_bnd) + x_lo_bnd, sig_bnd_999, '99.9%',fontsize=10);
    ax1.grid( linestyle= ':')
    plt.show()

#------------------------------------------------------------------------------
#                                HPLSTAT
#------------------------------------------------------------------------------
def hplstat(HPL,HPE,HAL,src_name):
        

    # size of VPL, which should be the same for VPE as well
    n = HPL.shape[1]

    # HPE
    j0        = np.floor(2.0*np.abs(HPE))#+1
    j0[j0>99] = 99;
    j0        = j0.astype(int)
    # HPL
    HPL       = HPL[0]
    k0        = np.floor(2.0*abs(HPL))#+1
    k0[k0>99] = 99;
    k0        = k0.astype(int)
    
    # initialize hpl_stat
    data      = np.zeros((100,100))
    diagonal  = np.zeros((100,1))
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
    
    err_bin       = np.arange(0.25,49.75+0.5,0.5)
    sig_bin       = np.arange(0.25,49.75+0.5,0.5)
    
    seconds       = n;
    sec_available = n;
    
    # determine the number of points and axis ranges
    n_pts         = np.sum(np.sum(data,axis= 0));
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
    
#    i0,j0  = np.where(data>0) 
    j0,i0     = np.where(data.T>0) 
    i0        = i0.astype(int)
    j0        = j0.astype(int)

    base1     = np.array([-0.5, 0.5 , 0.5, -0.5, -0.5, 0.5 , 0.5, -0.5])
    base2     = np.array([-0.5, -0.5, 0.5, 0.5 , -0.5, -0.5, 0.5, 0.5 ])

    df_STFD = pd.DataFrame(columns = ['X','Y','Z','c_idx'])
    for idx in range(np.size(i0)):
        z                        = np.log10(data[i0[idx],j0[idx]])
        col1                     = err_bin[j0[idx]] + base1*d_err_bin
        col2                     = sig_bin[i0[idx]] + base2*d_sig_bin
        vtx_mat                  = np.array([col1,col2,[0,0,0,0,z,z,z,z]])
        vtx_mat                  = vtx_mat.T
        c_idx                    = int(np.ceil(63*(np.log10(data[i0[idx],j0[idx]])/np.log10(z_up_bnd))))# + 1
        X                        = (vtx_mat[0,0] + vtx_mat[1,0])/2
        Y                        = (vtx_mat[1,1] + vtx_mat[2,1])/2
        Z                        = z
        df_STFD.loc[idx,'X']     = X
        df_STFD.loc[idx,'Y']     = Y
        df_STFD.loc[idx,'Z']     = Z
        df_STFD.loc[idx,'c_idx'] = c_idx


#==============================================================================

    fig_plotly = make_subplots(rows=1, cols=2, column_widths=[10, 0.5], subplot_titles=('Horizontal Performance ['+str(seconds)+' seconds]', ""),specs=[[{"secondary_y": False}, {"secondary_y": True}]] )

    matplt_cmap,color_bar = ColorMap()
    RGB_bar            = barra_RGB(color_bar) # to generate in  rgb(0.267004,0.004874,0.329415)
    plot_ColorBar(fig_plotly,RGB_bar,z_up_bnd)

    # determine availability and # of integrity failures
    i_diag1 = np.where(err_bin == HAL);
    i_diag2 = np.where(err_bin < HAL);
    i_diag3 = np.where(err_bin > HAL);
    cond1  = np.where(err_bin[j0] >= HAL)[0]
    cond2  = np.where(sig_bin[i0] < HAL)[0]
    i_fail1 = np.intersect1d(cond1, cond2)
    n_fail1 = np.sum( np.sum( np.diag( data[i0[i_fail1],j0[i_fail1]] ) ) ) - np.sum(diagonal[i_diag1])
   
    cond1   = np.where(err_bin[j0]/sig_bin[i0] >=1.0)[0]
    cond2   = np.where(err_bin[j0] < HAL)[0]
    i_fail2 = np.intersect1d(cond1, cond2)
    n_fail2 = np.sum( np.sum( np.diag( data[i0[i_fail2],j0[i_fail2]] ) ) ) - np.sum(diagonal[i_diag2])
    
    cond1   = np.where(err_bin[j0]/sig_bin[i0] >=1.0)[0]
    cond2   = np.where(sig_bin[i0] >= HAL)[0]
    i_fail3 = np.intersect1d(cond1, cond2)
    n_fail3 = np.sum( np.sum( np.diag( data[i0[i_fail3],j0[i_fail3]] ) ) ) - np.sum(diagonal[i_diag3])

    i_cont  = np.where(sig_bin[i0] >= HAL)[0]

    n_cont  = np.sum( np.sum( np.diag(data[i0[i_cont],j0[i_cont]])));
    
    #i_avail = find(err_bin(j) < HAL & sig_bin(i) < HAL);
    
    cond1   = np.where(err_bin[j0]/sig_bin[i0] < 1.0)[0]
    cond2   = np.where(sig_bin[i0] < HAL)[0]
    i_avail = np.intersect1d(cond1, cond2)
    n_avail = np.sum( np.sum( np.diag( data[i0[i_avail],j0[i_avail]] ) ) ) + np.sum(diagonal[i_diag2])
    
    txt_list = []
    txt_x    = []
    txt_y    = []
    
    # show the region of normal operation
    txt_list.append('Normal Operation'); txt_x.append(0.37*(HAL - x_lo_bnd) + x_lo_bnd); txt_y.append( 0.93*(HAL - y_lo_bnd) + y_lo_bnd)
    if  n_avail/epochs >= .999995:
        txt_list.append('> 99.999%');txt_x.append(0.37*(HAL - x_lo_bnd) + x_lo_bnd);txt_y.append(0.86*(HAL - y_lo_bnd) + y_lo_bnd)
    else:
        txt_list.append("{:.3f}".format(100.0*n_avail/epochs)+'%'); txt_x.append(0.37*(HAL - x_lo_bnd) + x_lo_bnd);txt_y.append(0.86*(HAL - y_lo_bnd) + y_lo_bnd)
   
    # outline the region of integrity failures (RECTANGULO ROJ0)
    add_polygon(fig_plotly,[HAL,HAL,x_up_bnd,x_up_bnd],[y_lo_bnd,HAL,HAL,y_lo_bnd],'rgb(255,114,111)')
    txt_list.append('HMI');txt_x.append(0.50*(x_up_bnd - HAL) + HAL);txt_y.append( 0.55*(HAL - y_lo_bnd) + y_lo_bnd)
    txt_list.append('epochs: '+ str(n_fail1));txt_x.append(0.50*(x_up_bnd - HAL) + HAL);txt_y.append( 0.45*(HAL - y_lo_bnd) + y_lo_bnd)
    
    # outline the region of HPL failures (TRIANGULO inferior)
    add_polygon(fig_plotly,[x_lo_bnd,HAL,HAL],[y_lo_bnd,HAL,y_lo_bnd],'rgb(255,204,203)')
    txt_list.append('MI'); txt_x.append( 0.67*(HAL - x_lo_bnd) + x_lo_bnd) , txt_y.append(0.35*(HAL - y_lo_bnd) + y_lo_bnd)
    txt_list.append('epochs: '+ str(n_fail2));txt_x.append(0.67*(HAL - x_lo_bnd) + x_lo_bnd) , txt_y.append(0.25*(HAL - y_lo_bnd) + y_lo_bnd)
    
    # outline the region of unavailability  (POLIGONO AMARILLO)
    add_polygon(fig_plotly,[x_lo_bnd, x_up_bnd, x_up_bnd, x_lo_bnd],[HAL, HAL, y_up_bnd, y_up_bnd],'rgb(253,255,143)')
    txt_list.append('System Unavailable');txt_x.append(0.50*(x_up_bnd - x_lo_bnd) + x_lo_bnd);txt_y.append(0.70*(y_up_bnd - HAL) + HAL )
    txt_list.append('epochs: '+ str(n_cont));txt_x.append( 0.50*(x_up_bnd - x_lo_bnd) + x_lo_bnd);txt_y.append(0.55*(y_up_bnd - HAL) + HAL )
    
    # outline the region where integrity failures and unavailability overlap (TRIANGULO superior)
    add_polygon(fig_plotly,[HAL, x_up_bnd, x_up_bnd],[HAL, y_up_bnd, HAL],'rgb(255,158,87)')
    txt_list.append('MI');txt_x.append(0.5*(x_up_bnd - HAL) + HAL);txt_y.append(0.325*(y_up_bnd - HAL) + HAL )
    txt_list.append('epochs: '+ str(n_fail3));txt_x.append(0.5*(x_up_bnd - HAL) + HAL);txt_y.append(0.175*(y_up_bnd - HAL) + HAL )
    
    add_text(fig_plotly,txt_list, txt_x,txt_y,'black')

    
    for k in df_STFD.index:
        add_patch  (fig_plotly,df_STFD.loc[k,'X'],df_STFD.loc[k,'Y'],RGB_bar[df_STFD.loc[k,'c_idx']])


    fig_plotly.update_xaxes(range=[x_lo_bnd, x_up_bnd], showgrid=True,title_text='Error [m]', gridwidth=1, gridcolor='LightPink', row=1, col=1)
    fig_plotly.update_yaxes(range=[y_lo_bnd, y_up_bnd], showgrid=True,title_text= r'$HPL_{'+src_name+'} [m]$', gridwidth=1, gridcolor='LightPink', row=1, col=1)
    fig_plotly.show()
    

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
vplstat(VPL.T,VPE,VAL1,VAL2,'EGNOS')
#ax.set_xlim3d(0, 50)
#ax.set_zlim3d(0, 10)
#ax.view_init(90, 270)
#plt.show()
