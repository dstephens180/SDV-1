
# LIBRARIES ----
import pandas as pd
import numpy as np
import janitor as jn
import pytimetk as tk
import yfinance as yf
import pytimetk as tk
import random
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio


from sdv.datasets.demo import download_demo
from sdv.sequential import PARSynthesizer

# this step is for metadata.visualize for GraphViz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'


# 0.0 DATA IMPORT ----
real_data, metadata = download_demo(
    modality='sequential',
    dataset_name='nasdaq100_2019'
)


# all unique symbols
ticker_array = real_data['Symbol'].unique()
ticker_list = ticker_array.tolist()

# random samples (just in case you want to filter later)
sample_tickers_list = random.sample(ticker_list, 20)


real_data.glimpse()
metadata.visualize()




# plot with matplotlib
plt.figure(figsize=(14, 7))

# Get unique symbols
symbols = real_data['Symbol'].unique()

for symbol in symbols:
    symbol_data = real_data[real_data['Symbol'] == symbol]
    plt.plot(symbol_data['Date'], symbol_data['Close'], label=symbol)

plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Prices Over Time')
plt.legend(title='Symbol')
plt.grid(True)
plt.show()




# 1.0 EDA ----
# IMPORTANT: this is multi-sequence data;, meaning, there are different sequences for each company: AMZN, GOOG, NFLX, AAPL
# PAR synthesizer is suited for multi-sequence data.

# 1.1 SEQUENCE KEY
# Sequence Key is the column to identify each stock ticker.
amd_sequence = real_data[real_data['Symbol'] == 'AMD']
amd_sequence

real_data['Symbol'].unique()

# 1.2 CONTEXT COLUMNS
# Context columns should never change.  So Sector & Industry are context columns.
real_data[real_data['Symbol'] == 'AMD']['Sector'].unique()
real_data[real_data['Symbol'] == 'AMD']['Industry'].unique()


# **The PAR Synthesizer learns sequence information based on the context.** It's important to identify these columns ahead of time.




# 2.0 CREATING A SYNTHESIZER ----
synthesizer = PARSynthesizer(
    metadata,
    context_columns=['Sector', 'Industry'],
    verbose=True)

synthesizer.fit(real_data)




# 2.1 GENERATE SYNTHETIC DATA ----
# passing 10 sequences to synthesize; synthesizer algorithmically determines how long to make each sequence.
synthetic_data = synthesizer.sample(num_sequences=10)
synthetic_data


synthetic_data[['Symbol', 'Industry']].groupby(['Symbol']).first().reset_index()



# 2.2 SAVE SYNTHESIZER & LOAD AGAIN
synthesizer.save('00_data/example_stock_synthesizer.pkl')

synthesizer = PARSynthesizer.load('00_data/example_stock_synthesizer.pkl')




# 3.0 PAR CUSTOMIZATION ----
# Use the epochs parameter to make a tradeoff between training time and data quality. 
# Higher epochs mean the synthesizer will train for longer, ideally improving the data quality.
# Use the enforce_min_max_values parameter to specify whether the synthesized data should always be within the same min/max ranges as the real data. 
# Toggle this to False in order to enable forecasting.
custom_synthesizer = PARSynthesizer(
    metadata,
    epochs=250,
    context_columns=['Sector', 'Industry'],
    enforce_min_max_values=True,
    verbose=False)

custom_synthesizer.fit(real_data)




# 4.0 SAMPLING OPTIONS ----

# 4.1 Specify Sequence Length ----
# By default, the synthesizer algorithmically determines the length of each sequence. 
# However, you can also specify a fixed, predetermined length.
custom_synthesizer.sample(num_sequences=3, sequence_length=2)





# To forecast values into the future, specify a longer sequence length.
# Now instead of ending at 2019 like the original data, the sequence goes until the end of 2022.
long_sequence = custom_synthesizer.sample(num_sequences=1, sequence_length=500)

long_sequence.head()
long_sequence.tail()



# 4.2 Conditional Sampling Using Context ----
scenario_context = pd.DataFrame(data={
    'Symbol': ['COMPANY-A', 'COMPANY-B', 'COMPANY-C', 'COMPANY-D', 'COMPANY-E'],
    'Sector': ['Technology']*3 + ['Consumer Services']*2,
    'Industry': ['Computer Manufacturing', 'Computer Software: Prepackaged Software', 
                 'Electronic Components', 'Catalog/Specialty Distribution', 'Catalog/Specialty Distribution']
})

scenario_context

# Sector array for later filtering
sector_filter = scenario_context['Sector'].unique()


custom_synthetic_data = custom_synthesizer.sample_sequential_columns(
    context_columns=scenario_context,
    sequence_length=None
)



custom_synthetic_data['Date'] = pd.to_datetime(custom_synthetic_data['Date'])


# bind all data
custom_combined_data = pd.concat([real_data, custom_synthetic_data], ignore_index=True)

# filter only the Sector columns in the custom scenario
filtered_df = custom_combined_data \
    .loc[custom_combined_data['Sector'].isin(sector_filter)]





# VISUALIZE ----
# Initialize the figure
custom_plot = go.Figure()

# Get unique symbols
symbols = filtered_df['Symbol'].unique()

# Add a trace for each symbol
for symbol in symbols:
    symbol_data = filtered_df[filtered_df['Symbol'] == symbol]
    custom_plot.add_trace(go.Scatter(x=symbol_data['Date'], y=symbol_data['Close'],
                             mode='lines', name=symbol))


# Update the layout
custom_plot.update_layout(
    title='Stock Prices Over Time',
    xaxis_title='Date',
    yaxis_title='Close Price',
    legend_title='Symbol',
    template='plotly_white',
    hovermode='x'
)

# Show the plot
custom_plot.show()













