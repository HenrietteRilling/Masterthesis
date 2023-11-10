# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:28:00 2023

@author: Henriette
"""

import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import cm2inch
import matplotlib.lines as mlines

# Sample data
data = {
    'Name': [
        'Tinnet Bro', 'Møllerup', 'Nedstrøms Haustrup Bæk', 'Tørring Kær',
        'ns Uldumkær', 'Åstedbro', 'Bredstenbro', 'Brestenbro',
        '500 m os Vorvadsbro', 'Højlund', 'Emborg Bro', 'Rye mølle - os bro',
        'Rye mølle - ns bro', 'Ved Resenbro', 'Tvilumbro', 'NS Tangeværket',
        'Bjerringbro', 'Ulstrup Bro', 'Åbro', 'Langå', 'Randers Bro'
    ],
    'Start time': [
        '14/04/2004', '08/03/2022', '02/03/2004', '06/03/2004',
        '09/11/2011', '01/01/1987', '01/12/2016', '22/10/2015',
        '01/01/1987', '23/01/2020', '23/11/2016', '30/01/2013',
        '31/01/2013', '18/10/2006', '01/01/1987', '01/01/2019',
        '11/02/2009', '01/01/1986', '02/06/2014', '11/02/2009', '06/12/2017'
    ],
    'End time': [
        '08/08/2023', '08/08/2023', '11/03/2010', '20/11/2009',
        '09/08/2023', '08/08/2023', '23/03/2022', '02/12/2021',
        '08/08/2023', '02/12/2021', '19/12/2022', '09/08/2023',
        '09/08/2023', '23/03/2022', '08/08/2023', '09/08/2023',
        '09/08/2023', '08/08/2023', '08/08/2023', '09/08/2023', '08/08/2023'
    ]
}

df = pd.DataFrame(data)
df['Start time'] = pd.to_datetime(df['Start time'], format='%d/%m/%Y')
df['End time'] = pd.to_datetime(df['End time'], format='%d/%m/%Y')

# Reverse the order of the DataFrame
df = df[::-1]

# Create a new figure
fig, ax = plt.subplots(figsize=(cm2inch(15, 10)))

# Create an invisible line with a label for data availability
invisible_line = mlines.Line2D([], [], color='blue', label='Data Availability', linewidth=5)


# Plot horizontal lines for each station's data availability
for index, row in df.iterrows():
    station_name = row['Name']
    start_time = row['Start time']
    end_time = row['End time']
    ax.hlines(station_name, start_time, end_time, colors='b', linewidth=5, label='Data availability')

# Set labels and title
ax.set_xlim(pd.Timestamp('1985-01-01'), pd.Timestamp('2024-05-31'))

# Set xtick labels fontsize to 34
ax.tick_params(axis='x', labelsize='medium', rotation=45)
# Set ytick labels fontsize to 34
plt.yticks(fontsize='medium')

ax.set_xlabel('Date', fontsize='large')

# Add a legend with a single entry for data availability
ax.legend(handles=[invisible_line], loc='lower left')

plt.tight_layout()
plt.savefig(r'C:\Users\henri\Documents\Universität\Masterthesis\Report\WL_data_range.png', dpi=600)
# Display the plot
plt.show()
