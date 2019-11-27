# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 21:50:31 2019

@author: 106300
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
import numpy as np
import matplotlib.patches as patches
import pickle
from collections import OrderedDict


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
    c_map        = cm.get_cmap(cmp_list[14]+'', 256) # ''= normal, '_r' = reversed
    color_BAR   = c_map(np.linspace(0, 1, 64))
    
    matlab      = 'inicial'    #-----------el de Matlab---------------
    matlab      = 'new'
    if matlab == 'inicial': 
        color_BAR_r = Reverse_ColorBar(color_BAR)
        c_map = ListedColormap(color_BAR_r)  
    return c_map,color_BAR


# other colors used in the plot
#color1 = [134/255 134/255 134/255] % - dark gray
color1 = (1,0.1,0.1,0.75)              # - red (original) 
#color2 = [174/255 174/255 174/255] % - middle gray
color2 = (1,0.55,0.55,0.75)            # - pink (original)
#color3 = [222/255 222/255 222/255] % - light gray
color3 = (1,1,0.5,0.75)                 #- yellow (original)
#color4 = color2                    % - middle gray
color4 = (1,.55,0.30,.75)   

# Getting back the objects:
file  =  open('objs.pkl', 'rb') 
datos = pickle.load(file)
df_stanford, x_lo_bnd, x_up_bnd, y_lo_bnd, y_up_bnd, HAL, epochs, n_cont, n_avail, n_fail1, i_diag1, n_fail2, n_fail3, seconds, z_lo_bnd,z_up_bnd = datos

"""
#==============================================================================
plt.figure(figsize=(12,10), dpi=70)
#-----------el de Matlab---------------
matplt_cmap = ListedColormap(SelColors)
#-----------uno de color---------------
#cmp_list    = [ 'viridis','Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
#                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
#                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
#matplt_cmap = cm.get_cmap(cmp_list[1]+'_r', 256) # ''= normal, '_r' = reversed
#SelColors   = matplt_cmap(np.linspace(0, 1, 64))
norm        = mpl.colors.Normalize(vmin=z_lo_bnd, vmax=z_up_bnd)   #-----aqui normalizamos
ax2         = plt.axes([.87-0.014, .1, .025, .8])
cb1         = mpl.colorbar.ColorbarBase(ax2, cmap = matplt_cmap, norm = norm, orientation = 'vertical')
ax1         = plt.axes([0.1-0.014, 0.1, .75, .8])
n_colores   = 64
for k in df_stanford.index:
    rect = patches.Rectangle( (df_stanford.loc[k,'X']-0.25, df_stanford.loc[k,'Y']-0.25),
                             0.5, 0.5, facecolor = SelColors[(n_colores-1)-df_stanford.loc[k,'c_idx']])
    ax1.add_patch(rect)    
ax1.axis([x_lo_bnd, x_up_bnd,y_lo_bnd, y_up_bnd])
cb1.set_label('Number of Points per Pixel')
ax1.set_aspect('equal', 'box')
#==============================================================================
"""
#==============================================================================
plt.figure(figsize=(12,10), dpi=70)
matplt_cmap,color_bar = ColorMap()
norm        = mpl.colors.Normalize(vmin=z_lo_bnd, vmax=z_up_bnd)   #-----aqui normalizamos

ax1         = plt.axes([0.1-0.014, 0.1, .75, .8])
n_colores   = 64
for k in df_stanford.index:
    rect = patches.Rectangle( (df_stanford.loc[k,'X']-0.25, df_stanford.loc[k,'Y']-0.25),
                             0.5, 0.5, facecolor = color_bar[df_stanford.loc[k,'c_idx']])
    ax1.add_patch(rect)    
ax1.axis([x_lo_bnd, x_up_bnd,y_lo_bnd, y_up_bnd])
ax1.set_aspect('equal', 'box')

ax2         = plt.axes([.87-0.014, .1, .025, .8])
cb1         = mpl.colorbar.ColorbarBase(ax2, cmap = matplt_cmap, norm = norm, orientation = 'vertical')
cb1.set_label('Number of Points per Pixel')

#==============================================================================


# show the region of normal operation
HT = ax1.text(0.37*(HAL - x_lo_bnd) + x_lo_bnd, 0.93*(HAL - y_lo_bnd) + y_lo_bnd, 'Normal Operation', rotation=0)
print('original',(0.37*(HAL - x_lo_bnd) + x_lo_bnd, 0.93*(HAL - y_lo_bnd) + y_lo_bnd))
print(HT.get_position())
HT.set_fontsize(30)

ax1.text(0.37*(HAL - x_lo_bnd) + x_lo_bnd, 0.93*(HAL - y_lo_bnd) + y_lo_bnd, 'Normal Operation', rotation=0, horizontalalignment =  'center',fontsize=10)
if  n_avail/epochs >= .999995:
    ax1.text(0.37*(HAL - x_lo_bnd) + x_lo_bnd, 0.86*(HAL - y_lo_bnd) + y_lo_bnd, '> 99.999%', rotation=0)
else:
    ax1.text(0.37*(HAL - x_lo_bnd) + x_lo_bnd, 0.86*(HAL - y_lo_bnd) + y_lo_bnd, "{:.2f}".format(100.0*n_avail/epochs)+'%',rotation=0)
        

# outline the region of integrity failures (RECTANGULO ROJ0)
coor = np.array([[HAL,HAL,x_up_bnd,x_up_bnd],[y_lo_bnd,HAL,HAL,y_lo_bnd]])
ax1.add_patch(patches.Polygon(coor.T, linewidth=0,facecolor=color1,zorder=0))
HT = ax1.text(0.50*(x_up_bnd - HAL) + HAL, 0.55*(HAL - y_lo_bnd) + y_lo_bnd, 'HMI')
HT = ax1.text(0.50*(x_up_bnd - HAL) + HAL, 0.45*(HAL - y_lo_bnd) + y_lo_bnd, 'epochs: '+ str(n_fail1))


# outline the region of HPL failures (TRIANGULO inferior)
coor = np.array([[x_lo_bnd,HAL,HAL],[y_lo_bnd,HAL,y_lo_bnd]])
ax1.add_patch(patches.Polygon(coor.T, linewidth=1,facecolor=color2,edgecolor='black',zorder=0))
ax1.text(0.67*(HAL - x_lo_bnd) + x_lo_bnd,0.35*(HAL - y_lo_bnd) + y_lo_bnd, 'MI')
HT = ax1.text(0.67*(HAL - x_lo_bnd) + x_lo_bnd,0.25*(HAL - y_lo_bnd) + y_lo_bnd, 'epochs: '+ str(n_fail2))
 
# outline the region of unavailability  (POLIGONO AMARILLO)
coor = np.array([[x_lo_bnd, x_up_bnd, x_up_bnd, x_lo_bnd],[HAL, HAL, y_up_bnd, y_up_bnd]])
ax1.add_patch(patches.Polygon(coor.T, linewidth=1,facecolor=color3,edgecolor='black',zorder=0))
ax1.text     (0.50*(x_up_bnd - x_lo_bnd) + x_lo_bnd,0.70*(y_up_bnd - HAL) + HAL, 'System Unavailable')
HT = ax1.text(0.50*(x_up_bnd - x_lo_bnd) + x_lo_bnd,0.55*(y_up_bnd - HAL) + HAL, 'epochs: '+ str(n_cont))

# outline the region where integrity failures and unavailability overlap (TRIANGULO superior)
coor = np.array([[HAL, x_up_bnd, x_up_bnd],[HAL, y_up_bnd, HAL]])
ax1.add_patch(patches.Polygon(coor.T, linewidth=1,facecolor=color4,edgecolor='black',zorder=0))
ax1.text     (0.5*(x_up_bnd - HAL) + HAL,0.325*(y_up_bnd - HAL) + HAL, 'MI')
HT = ax1.text(0.5*(x_up_bnd - HAL) + HAL,0.175*(y_up_bnd - HAL) + HAL, 'epochs: '+ str(n_fail3))


ax1.set_xlabel('Error [m]')

src_name= 'EGNOS'
ax1.set_ylabel(r'$HPL_{'+src_name+'} [m]$')

ax1.set_title('Horizontal Performance ['+str(seconds)+' seconds]')

#    plt.axis([x_lo_bnd, x_up_bnd,y_lo_bnd,y_up_bnd])
ax1.grid( linestyle= ':')
plt.show()




#-----------------------------------------------------------------------------
#                         EN COLOR
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#        OTRO COLOR 2A
#-----------------------------------------------------------------------------



#x      = [5,6,3,4,5,6,7,8,9,3,4,7]
#y      = [2,4,3,6,8,1,3,5,6,7,5,5]
#
#s      = [123,12,423,234,234,24,123,124,123,234,54,23]
#colors = [1,2,4,5,6,7,8,3,4,6,7,63]


##------------------------------------------------------------------------------
#
#
##subs = 20
##fig = plt.figure(figsize=(8,8))
##ax1 = plt.subplot2grid((subs,subs), (0,0), rowspan=subs, colspan=subs-1)
##
##ax1.scatter(x,y,s = 100,c=colors,cmap = newcmp,marker ='s')
##ax1.axis([0,10,0,10])
##
##ax2 = plt.subplot2grid((subs,subs), (0,subs-1), rowspan=subs)
##cb1 = mpl.colorbar.ColorbarBase( ax2,cmap=newcmp,
##                                norm=norm,
##                                orientation='vertical')
##cb1.set_label('Some Units')
##plt.show()
##------------------------------------------------------------------------------
#
#
#fig = plt.figure(figsize=(9,8))
#ax1 = plt.axes([0.1-0.014, 0.1, .75, .8])
#
##ax1.scatter(x,y,s = 100,c=colors,cmap = newcmp,marker ='s')
#ax1.add_patch(patches.Rectangle((1, 1), 1, 1, linewidth=1,facecolor=[0.0247 , 0.0247 , 0.0247 , 1],edgecolor='r'))
#
#coor = np.array([[35,50,50],[35,35,50]])
#
#ax1.add_patch(patches.Polygon(coor.T, linewidth=10,facecolor=[0.3 ,0.4 , 0.9 , 1],edgecolor='r',zorder=0))
#
#ax1.axis([0,50,0,50])
##ax1.set_aspect('equal', 'box')
#
#ax2 =  plt.axes([.87-0.014, .1, .025, .8])
#
#cb1 = mpl.colorbar.ColorbarBase( ax2,cmap=newCMP,
#                                norm=norm,
#                                orientation='vertical')
#
#cb1.set_label('Some Units')
##plt.tight_layout()  
#src_name= 'EGNOS'
#ax1.set_xlabel('Error [m]')
##ax1.set_ylabel(r'$\alpha_i > \beta_i$' )
#ax1.set_ylabel(r'$HPL_{'+src_name+'} [m]$')
#ax1.text(2, 6, r'an equation: $E=mc_2$', fontsize=15)
#ax1.text(2, 4, '{\fontsize{14pt}Normal Operation}', fontsize=15)
#HT = ax1.text(10, 10, 'HMI')
#print(ax1.get_position('HT'))
#HT.set_position((11,11))
#print(ax1.get_position('HT'))
#ax1.grid( linestyle= ':')
#plt.show()
#
#
