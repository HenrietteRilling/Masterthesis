# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 10:22:00 2024

@author: Henriette
"""
import glob
import os
import re
from webcolors import hex_to_rgb_percent

def html2rgb(match):
    htmlcode=match[0].replace('{', '#').replace('}', '')
    rgb_tuple=hex_to_rgb_percent(htmlcode)
    red=str(round(float(rgb_tuple.red.strip('%'))/100,4))
    green=str(round(float(rgb_tuple.green.strip('%'))/100,4))
    blue=str(round(float(rgb_tuple.blue.strip('%'))/100,4))
    return '{'+red+','+green+','+blue+'}'

if __name__=="__main__":
    #get paths to folder and txt file with tables
    filepath=r'C:\Users\henri\Documents\Universität\Masterthesis\Report\Table_latex_conversion'
    alltablepaths=glob.glob(os.path.join(filepath, "LSTM_all_PI.txt"))
    for path in alltablepaths:
        #read table data
        with open(path, 'r') as file:
            content = file.read()
        
        #replace HTML prompt with rgb
        content=content.replace('HTML', 'rgb')
        
        #define pattern to find 6 digit html code
        html_pattern=r'\{([0-9A-Fa-f]{6})\}'
        modified_content=re.sub(html_pattern, html2rgb, content)
    
    # Write the modified content back to a new file
        savepath=os.path.join(filepath, f'latex_rgb_{os.path.basename(path)}')
        with open(savepath, 'w') as file:
            file.write(modified_content)
    