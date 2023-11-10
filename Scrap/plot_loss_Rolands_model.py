# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 17:41:14 2023

@author: Henriette
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

losses=pd.read_csv(r'C:\Users\henri\Documents\Universität\Masterthesis\Masterthesis\results\losslog.csv', sep=';', header=None)

epoch=np.arange(len(losses)+1)
epoch=epoch[1:]

plt.figure()
plt.plot(epoch,losses[0], label='train')
plt.plot(epoch,losses[1], label='validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('MSE loss')

fig, ax=plt.subplots(2,1, sharex=True)
ax[0].plot(epoch, losses[0], label='train')
ax[0].plot(epoch, losses[1], label='validation')
ax[0].set_ylabel('MSE loss', fontsize=12)
ax[0].legend(fontsize=12)
ax[1].plot(epoch-0.5, losses[0], label='train shifted')
ax[1].plot(epoch, losses[1], label='validation')
ax[1].set_ylabel('MSE loss', fontsize=12)
ax[1].legend(fontsize=12)
ax[1].set_xlabel('Epoch', fontsize=12)


# losses2=pd.read_csv(r'C:\Users\henri\Documents\Universität\Masterthesis\Example Roland\results\losslog.csv', sep=';', header=None)

# plt.figure()
# plt.plot(losses2[0], label='train')
# plt.plot(losses2[1], label='validation')
# plt.legend()