# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 09:08:08 2023

@author: Henriette
"""

import pandas as pd
import numpy as np

from Data_loader import get_WL_data, get_prcp_data

#Load data
WL, _, station_name_to_id, _ = get_WL_data(r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\Data_WL')
SVK, DMI=get_prcp_data(r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\SVK', r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\DMI_Climate_Data_prcp')

