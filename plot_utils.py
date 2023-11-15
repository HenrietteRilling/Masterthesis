# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 18:34:24 2023

@author: Henriette
"""
import matplotlib.dates as mdates
from datetime import datetime, timedelta


# Thesis max size is 15.0 x 24.14 cm. Borrowed from https://zenodo.org/records/6726556 plot_utils
def cm2inch(*tupl: tuple):
    """
    Converts from cm to inches when defining figure sizes. Normal A4 page is 21 x 29.7 cm.
    OBS! Thesis uses max_width=15cm and max_height=24.4cm

    Args
     tupl: Tuple containing (width, height) in cm
    """
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
    
def datetime_xaxis(start, end):
    # Generate datetime range with hourly frequency
    date_range = [start + timedelta(hours=i) for i in range(int((end - start).total_seconds() / 3600) + 1)]
    
    # Convert datetime objects to strings for plotting
    date_strings = [date.strftime('%Y-%m-%d %H') for date in date_range]
    return date_strings

