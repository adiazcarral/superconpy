#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:58:26 2023

@author: mroitegui
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:47:15 2023

@author: mroitegui
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 15:59:20 2022

@author: angel
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
   
features=[
                            'Mend', 
                            'Mass',
                            'EN',
                            'Maradius',
                            'Mval',
                            'Mtc',
                            'Mfie',
                            'Mec',
                            'Smix',
                            'delta',
                        ]
vecs=[
      '1s','2s','2p','3s','3p','3d',
                              # '4s',
                              '4p','4d','4f','5s','5p','5d','5f','6s','6p',
                              # '6d',
                              '6f',
                              # '7s',
                              '7p'
      ]

################################
################################

# create some data to use for the plot
dt = 0.001
t = np.arange(0.0, 10.0, dt)
r = np.exp(-t[:1000]/0.05)               # impulse response
x = np.random.randn(len(t))
s = np.convolve(x, r)[:len(x)]*dt  # colored noise

fig = plt.figure(figsize=(9, 4),facecolor='white')
ax = fig.add_subplot(121)
# the main axes is subplot(111) by default
plt.plot(t, s)
plt.axis([0, 1, 1.1*np.amin(s), 2*np.amax(s)])
plt.xlabel('time (s)')
plt.ylabel('current (nA)')
plt.title('Subplot 1: \n Gaussian colored noise')

# this is an inset axes over the main axes
inset_axes = inset_axes(ax, 
                    width="50%", # width = 30% of parent_bbox
                    height=1.0, # height : 1 inch
                    loc=1)
n, bins, patches = plt.hist(s, 400)
#plt.title('Probability')
plt.xticks([])
plt.yticks([])


plt.tight_layout()
plt.show()
####################
###################
# END OF THE CODE #
###################
