#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:48:00 2023

@author: mroitegui

"""
import numpy as np
import matplotlib.pyplot as plt

#Load data
allr2=np.load('allr2.npy')
r2all=np.load('r2all.npy')
r2us=np.load('r2us.npy')
r2vec=np.load('r2vec.npy')
aller=np.load('aller.npy')
user=np.load('user.npy')
vecer=np.load('vecer.npy')
Mus=np.load('Mus.npy')
M_u_er=np.load('M_u_er.npy')
M_v_er=np.load('M_v_er.npy')
M_a_er=np.load('M_a_er.npy')
Mvec=np.load('Mvec.npy')
Mall=np.load('Mall.npy')

#Define x ticks
ticks=['Ec','new','other']


#Create the plot
fig, ax1 = plt.subplots()

color2 = 'tab:red'
# ax1.set_xlabel('time (s)')
ax1.set_ylabel('$R^2$', color=color2,fontsize=18)
ax1.scatter(ticks,[r2vec,r2us,r2all],color=color2)
plt.xticks(fontsize=18)
plt.yticks(fontsize=15)
plt.ylim((0.865, 0.935))
ax1.errorbar(ticks,[r2vec,r2us,r2all],yerr=[aller,user,vecer],fmt="o",color=color2,markersize=8, capsize=5)
# ax1.plot(t, data1, color=color)
ax1.tick_params(axis='y', labelcolor=color2)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('MAE (K)', color=color,fontsize=18) 
plt.yticks(fontsize=15)
plt.ylim((4.05, 5.75))
ax2.scatter(ticks,[Mvec,Mus,Mall],color=color)# we already handled the x-label with ax1
ax2.errorbar(ticks,[Mvec,Mus,Mall],yerr=[M_v_er,M_u_er,M_a_er],fmt="o",color=color,markersize=8, capsize=5)
# ax2.plot(t, data2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()



