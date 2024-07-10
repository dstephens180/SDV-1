
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




# 0.1 DOWNLOAD HISTORICAL DATA FROM YF ----
raw_df = yf.download(ticker_list, start='2023-01-01', end='2023-12-31', progress=False)


raw_data_prepared = raw_df \
    .stack() \
    .reset_index(level=1, names=["", "Symbol"]) \
    .sort_values(by=["Symbol", "Date"]) \
    .reset_index() \
    .drop_duplicates(subset=["Symbol", "Date"])

raw_data_prepared.glimpse()

brmn_data = raw_data_prepared[raw_data_prepared['Symbol'] == 'BMRN']


# plot with matplotlib
plt.figure(figsize=(14, 7))

# Get unique symbols
symbols = raw_data_prepared['Symbol'].unique()

for symbol in symbols:
    symbol_data = raw_data_prepared[raw_data_prepared['Symbol'] == symbol]
    plt.plot(symbol_data['Date'], symbol_data['Close'], label=symbol)

plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Prices Over Time')
plt.legend(title='Symbol')
plt.grid(True)
plt.show()





# 0.2 PREPARE RAW DATA ----

unique_tickers_df = real_data[['Symbol', 'Sector', 'Industry', 'MarketCap']] \
    .drop_duplicates()


# join data
merged_df = raw_data_prepared.merge(unique_tickers_df, on='Symbol', how='left')

# drop NA's
merged_df = merged_df.dropna(subset=['Sector', 'Industry'])

merged_prepared_df = merged_df[['Symbol', 'Date', 'Open', 'Close', 'Volume', 'MarketCap', 'Sector', 'Industry']]
merged_prepared_df.glimpse()


# visualize (convert back to object for testing)
merged_prepared_df['Date'] = merged_prepared_df['Date'].astype(str)



# plot with matplotlib
plt.figure(figsize=(14, 7))

# Get unique symbols
symbols = merged_prepared_df['Symbol'].unique()

for symbol in symbols:
    symbol_data = merged_prepared_df[merged_prepared_df['Symbol'] == symbol]
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
amd_sequence = merged_prepared_df[merged_prepared_df['Symbol'] == 'AMD']
amd_sequence



# 1.2 CONTEXT COLUMNS
# Context columns should never change.  So Sector & Industry are context columns.
merged_prepared_df[merged_prepared_df['Symbol'] == 'AMD']['Sector'].unique()
merged_prepared_df[merged_prepared_df['Symbol'] == 'AMD']['Industry'].unique()


# **The PAR Synthesizer learns sequence information based on the context.** It's important to identify these columns ahead of time.




# 2.0 CREATING A SYNTHESIZER ----
synthesizer = PARSynthesizer(
    metadata,
    context_columns=['Sector', 'Industry'],
    verbose=True)

synthesizer.fit(merged_prepared_df)




# 2.1 GENERATE SYNTHETIC DATA ----
# passing 10 sequences to synthesize; synthesizer algorithmically determines how long to make each sequence.
synthetic_data = synthesizer.sample(num_sequences=10)
synthetic_data.head()


synthetic_data['Date'] = pd.to_datetime(synthetic_data['Date'])

synthetic_data \
    .plot_timeseries(
        color_column = 'Symbol',
        date_column  = 'Date',
        value_column = 'Close',
        engine = 'plotly',
        smooth = False,
    )



# bind with real data
combined_data = pd.concat([merged_prepared_df, synthetic_data], ignore_index=True)

combined_data['Date'] = pd.to_datetime(combined_data['Date'])


# filter technology sector only (real & synthetic)
technology_df = combined_data[combined_data['Sector'] == 'Consumer Services']


# Initialize the figure
fig = go.Figure()

# Get unique symbols
symbols = technology_df['Symbol'].unique()

# Add a trace for each symbol
for symbol in symbols:
    symbol_data = technology_df[technology_df['Symbol'] == symbol]
    fig.add_trace(go.Scatter(x=symbol_data['Date'], y=symbol_data['Close'],
                             mode='lines', name=symbol))

# Update the layout
fig.update_layout(
    title='Stock Prices Over Time',
    xaxis_title='Date',
    yaxis_title='Close Price',
    legend_title='Symbol',
    template='plotly_white',
    hovermode='x'
)

# Show the plot
fig.show()





# The synthesizer generates entirely new sequences in the same format as the real data. 
# Each sequence represents an entirely new company based on the overall patterns from the dataset. 
# They do not map or correspond to any real company.
# For example, fictitious company AAAA is a generic Consumer Electronics/Video Chains company and AAAB is a Business Services company.
# A full list of our synthetic companies is shown below.
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
    enforce_min_max_values=False,
    verbose=True)

custom_synthesizer.fit(merged_prepared_df)




# 4.0 SAMPLING OPTIONS ----

# 4.1 Specify Sequence Length ----
# By default, the synthesizer algorithmically determines the length of each sequence. 
# However, you can also specify a fixed, predetermined length.
custom_df = custom_synthesizer.sample(num_sequences=5, sequence_length=None)
custom_df[['Symbol', 'Sector', 'Industry']] \
    .groupby(['Symbol']) \
    .first() \
    .reset_index()

custom_df['Date'] = pd.to_datetime(custom_df['Date'])

custom_df \
    .plot_timeseries(
        color_column = 'Symbol',
        date_column  = 'Date',
        value_column = 'Close',
        engine = 'plotly',
        smooth = False,
    )







# To forecast values into the future, specify a longer sequence length.
# Now instead of ending at 2019 like the original data, the sequence goes until the end of 2022.
long_sequence = custom_synthesizer.sample(num_sequences=5, sequence_length=250)

long_sequence['Date'] = pd.to_datetime(long_sequence['Date'])

long_sequence \
    .plot_timeseries(
        color_column = 'Symbol',
        date_column  = 'Date',
        value_column = 'Close',
        engine = 'plotly',
        smooth = False,
    )



# 4.2 Conditional Sampling Using Context ----
# You can pass in context columns and allow the PAR synthesizer to simulate the sequence based on those values.
# Let's start by creating a scenario with 2 companies in the Technology sector and 3 others in the Consumer Services sector. 
# Each row corresponds to a new sequence that we want to synthesize.


scenario_context = pd.DataFrame(data={
    'Symbol': ['COMPANY-A', 'COMPANY-B', 'COMPANY-C', 'COMPANY-D', 'COMPANY-E'],
    'Sector': ['Technology']*2 + ['Consumer Services']*3,
    'Industry': ['Computer Manufacturing', 'Computer Software: Prepackaged Software',
                 'Hotels/Resorts', 'Restaurants', 'Clothing/Shoe/Accessory Stores']
})

# Sector array for later filtering
sector_filter = scenario_context['Sector'].unique()


# Now we can simulate this scenario using our trained synthesizer.
custom_synthetic_data = custom_synthesizer.sample_sequential_columns(
    context_columns=scenario_context,
    sequence_length=250
)



custom_synthetic_data['Date'] = pd.to_datetime(custom_synthetic_data['Date'])


# bind all data
custom_combined_data = pd.concat([merged_prepared_df, custom_synthetic_data], ignore_index=True)

# filter only the Sector columns in the custom scenario
filtered_df = custom_combined_data \
    .loc[custom_combined_data['Sector'].isin(sector_filter)]


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













