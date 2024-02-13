# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 22:48:33 2024

@author: Henriette
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors

from plot_utils import cm2inch

filepath=r'C:\Users\henri\Documents\Universität\Masterthesis\Report\test3.csv'
df=pd.read_csv(filepath)
rmse=df.copy().drop(columns=df.columns[-6:])
mlist=['v', 'o', '*', 'None']
lstyles=['solid','dotted','dashed', 'dashdot', (0, (1, 10))]
clist=['darkviolet','blue', 'green', 'orange']
x=[  1,  12,  24,  48, 168, 672]
train_h=np.unique(rmse['TH [h]'])
windows=np.unique(rmse['W [h]'])
model=np.unique(rmse['Name'])



fig, axes =plt.subplots(2,2, figsize=cm2inch((15, 9)), sharex=True)
axes=axes.flatten()

# Create legend entries for linestyles
for j, th in enumerate(train_h):
    axes[0].plot([], [], linestyle=lstyles[j], color='black', label=f'TH: {th} h')
    
for i, w in enumerate(windows):
    ax=axes[i]

    for m, mod in enumerate(model): 
    
        for j, th in enumerate(train_h):
        
            if mod =='FFNN-P':
                plt_df=rmse[(rmse['Name']==mod)&(rmse['W [h]']==10) & (rmse['TH [h]']==th)].to_numpy()
            else: plt_df=rmse[(rmse['Name']==mod)&(rmse['W [h]']==w) & (rmse['TH [h]']==th)].to_numpy()
            
            
            if j==0 and i==0:
                ax.plot(x,plt_df[0][3:], linestyle=lstyles[j], marker=mlist[m], color=clist[m], ms=3, lw=1, label=f'{mod}')
            else:
                ax.plot(x,plt_df[0][3:], linestyle=lstyles[j], marker=mlist[m], color=clist[m], ms=3, lw=1)
    # ax.set_xlim(-5, 50)
    ax.set_ylim(-0.01,0.6)
    ax.set_title(f'W={w} h', loc='left')
    plt.legend()


fig.delaxes(axes[3])


fig.legend(loc='upper center', ncol=2, fontsize='small', frameon=True, fancybox=False, edgecolor='black', bbox_to_anchor=(0.76, 0.5), markerscale=2)  
fig.text(0.02, 0.5, 'RMSE [m]', va='center', rotation='vertical', fontsize='large')
fig.supxlabel('Imputation horizon [h]')
    
plt.subplots_adjust(left= 0.06, bottom=0,right=0.96, top=1.0, hspace=0.2)
#adjust space tight layout is taking in windows canva, neede that legend on top and label in bottom are shown. 
plt.tight_layout(rect=[0.06, 0 ,0.96, 1.0],pad=0.3) #rect: [left, bottom, right, top]
plt.savefig(r'C:\Users\henri\Documents\Universität\Masterthesis\Report\overview_final.png', dpi=600)
