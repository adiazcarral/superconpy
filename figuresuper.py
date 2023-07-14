#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:41:22 2023

@author: mroitegui
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
########data
Xg_total=pd.read_csv('/work/mroitegui/Superconductivity/xgboost_shap',index_col=0)
RF_total=pd.read_csv('/work/mroitegui/Superconductivity/RF_shap',index_col=0)
MLP_total=pd.read_csv('/work/mroitegui/Superconductivity/MLP_shap',index_col=0)


Xg_vecs=Xg_total.loc[['1s','2s','2p','3s','3p','3d',
                        # '4s',
                        '4p','4d','4f','5s','5p','5d','5f','6s','6p',
                        # '6d',
                        '6f',
                        # '7s',
                        '7p']].copy()

Xg_total=Xg_total.drop(['1s','2s','2p','3s','3p','3d',
                            # '4s',
                            '4p','4d','4f','5s','5p','5d','5f','6s','6p',
                            # '6d',
                            '6f',
                            # '7s',
                            '7p'])

RF_vecs=RF_total.loc[['1s','2s','2p','3s','3p','3d',
                        # '4s',
                        '4p','4d','4f','5s','5p','5d','5f','6s','6p',
                        # '6d',
                        '6f',
                        # '7s',
                        '7p']].copy()

RF_total=RF_total.drop(['1s','2s','2p','3s','3p','3d',
                            # '4s',
                            '4p','4d','4f','5s','5p','5d','5f','6s','6p',
                            # '6d',
                            '6f',
                            # '7s',
                            '7p'])
MLP_vecs=MLP_total.loc[['1s','2s','2p','3s','3p','3d',
                        # '4s',
                        '4p','4d','4f','5s','5p','5d','5f','6s','6p',
                        # '6d',
                        '6f',
                        # '7s',
                        '7p']].copy()

MLP_total=MLP_total.drop(['1s','2s','2p','3s','3p','3d',
                            # '4s',
                            '4p','4d','4f','5s','5p','5d','5f','6s','6p',
                            # '6d',
                            '6f',
                            # '7s',
                            '7p'])

features=[r'$\bar {P}_m $','M',r'$ \chi $', r'$ \bar{R}_{a} $',r'$ \bar{e}_{v} $',r'$ \bar{\kappa} $ ',r'$ \bar{I} $ ',r'$ \bar{\sigma} $', r'$ \Delta S_{mix} $', r'$ \delta $',r'$n_{e}$' ]
# print (features)
Xg_total=Xg_total.rename(columns={'0':'Xgboost'})
plot_total=Xg_total.copy()
plot_total['RF']=RF_total['0'].copy()
plot_total['MLP']=MLP_total['0'].copy()
suma =plot_total.sum()

# plot_total.append('suma')
plot_total['RF']=plot_total['RF']/suma.loc['RF']
plot_total['Xgboost']=plot_total['Xgboost']/suma.loc['Xgboost']
plot_total['MLP']=plot_total['MLP']/suma.loc['MLP']
plot_total.plot(ylabel='Mean Shap values',kind='bar')

plt.show()
Xg_vecs=Xg_vecs.rename(columns={'0':'Xgboost'})
plot_vecs=Xg_vecs.copy()
plot_vecs['RF']=RF_vecs['0'].copy()
plot_vecs['MLP']=MLP_vecs['0'].copy()
sumav=plot_vecs.sum()
plot_vecs['RF']=plot_vecs['RF']/suma.loc['RF']
plot_vecs['Xgboost']=plot_vecs['Xgboost']/suma.loc['Xgboost']
plot_vecs['MLP']=plot_vecs['MLP']/suma.loc['MLP']

plot_vecs.plot(ylabel='Mean Shap values',kind='bar')
plt.show()

plot_total=plot_total.set_axis(features)

# fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(12, 12))
# plot_total.plot(ylabel='Mean Shap values',kind='bar',ax=axes[0])
# plot_vecs.plot(ylabel='Mean Shap values',kind='bar',ax=axes[1])
# plt.show()


fig, ax1 = plt.subplots()

# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.39, 0.35, 0.44, 0.5]
ax2 = fig.add_axes([left, bottom, width, height])
# ax2.set_xticks(features,rotation = 45)
# plot_total.plot(rot=0,ylabel='Mean Shap values',kind='bar',width=0.65,ax=ax1,fontsize=18).set_ylabel('% SHAP',fontdict={'fontsize':18})
plot_total.plot(ylabel='Mean Shap values',kind='bar',width=0.65,ax=ax1,fontsize=18).set_ylabel('% SHAP',fontdict={'fontsize':18})


plot_vecs.plot(kind='bar',width=0.8,ax=ax2,legend=False)








# plt. subtitle("Plotting multiple Graphs")
# plt.show()
# def add_subplot_axes(ax,rect,facecolor='w'):
#     fig = plt.gcf()
#     box = ax.get_position()
#     width = box.width
#     height = box.height
#     inax_position  = ax.transAxes.transform(rect[0:2])
#     transFigure = fig.transFigure.inverted()
#     infig_position = transFigure.transform(inax_position)    
#     x = infig_position[0]
#     y = infig_position[1]
#     width *= rect[2]
#     height *= rect[3]  # <= Typo was here
#     subax = fig.add_axes([x,y,width,height],facecolor=facecolor)  # matplotlib 2.0+
#     # subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
#     x_labelsize = subax.get_xticklabels()[0].get_size()
#     y_labelsize = subax.get_yticklabels()[0].get_size()
#     x_labelsize *= rect[2]**0.5
#     y_labelsize *= rect[3]**0.5
#     subax.xaxis.set_tick_params(labelsize=x_labelsize)
#     subax.yaxis.set_tick_params(labelsize=y_labelsize)
#     return subax
    
# def example1():
#     fig = plt.figure(figsize=(10,10))
#     ax = fig.add_subplot(111)
#     rect = [0.1,0.1,1.7,1.7]
#     ax1 = add_subplot_axes(ax,rect)
#     ax2 = add_subplot_axes(ax1,rect)
   
#     plt.show()

# def example2():
#     fig = plt.figure(figsize=(10,10))
#     axes = []
#     subpos = [0.2,0.6,0.3,0.3]
#     x = np.linspace(-np.pi,np.pi)
#     for i in range(4):
#         axes.append(fig.add_subplot(2,2,i))
#     for axis in axes:
#         axis.set_xlim(-np.pi,np.pi)
#         axis.set_ylim(-1,3)
#         axis.plot(x,np.sin(x))
#         subax1 = add_subplot_axes(axis,subpos)
#         subax2 = add_subplot_axes(subax1,subpos)
#         subax1.plot(x,np.sin(x))
#         subax2.plot(x,np.sin(x))
# if __name__ == '__main__':
#     example2()
#     plt.show()

# example1()