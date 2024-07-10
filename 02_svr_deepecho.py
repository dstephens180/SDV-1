
# LIBRARIES ----
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
import pytimetk as tk
import pandas as pd
import numpy as np

from deepecho import PARModel
from sdv.sequential import PARSynthesizer

# this step is for metadata.visualize for GraphViz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'





# 1.0 DATA IMPORT ----
combined_listings = pd.read_csv("00_data/combined_listings_svr.csv")
combined_calendar = pd.read_csv("00_data/track_combined_calendar_svr.csv")
reservation_units_daily = pd.read_csv("00_data/reservations_units_daily_svr.csv")


# Data prep
combined_calendar['date'] = pd.to_datetime(combined_calendar['date'])
combined_calendar['listing_id'] = combined_calendar['listing_id'].astype(str)

combined_calendar_augmented = (
    combined_calendar.augment_timeseries_signature(date_column = "date")
)

combined_calendar_augmented.glimpse()



# reduce data
cc_prepared = combined_calendar_augmented[['listing_id', 'date', 'rate', 'available', 'date_index_num', 'date_year', 'date_half', 'date_quarter', 'date_month', 'date_yweek', 'date_wday', 'date_mday', 'date_yday']]



# Define data types for all the columns
data_types = {
    'date_index_num': 'categorical',
    'date_year': 'categorical',
    'date_half': 'categorical',
    'date_quarter': 'categorical',
    'date_month': 'categorical',
    'date_yweek': 'categorical',
    'date_wday': 'categorical',
    'date_mday': 'categorical',
    'date_yday': 'categorical',
    'rate': 'continuous',
    'available': 'count',
}


# unique listings
unique_listings = cc_prepared['listing_id'].unique()
first_n_listings = unique_listings[0:2]


# filter for the n listings
calendar_filtered = cc_prepared[cc_prepared['listing_id'].isin(first_n_listings)]




# VISUALIZE ----
custom_plot = go.Figure()


# Add a trace for each symbol
for listing_id in first_n_listings:
    symbol_data = calendar_filtered[calendar_filtered['listing_id'] == listing_id]
    custom_plot.add_trace(go.Scatter(x=symbol_data['date'], y=symbol_data['rate'],
                             mode='lines', name=listing_id))


# Update the layout
custom_plot.update_layout(
    title='Daily Rate by Listing ID',
    xaxis_title='Date',
    yaxis_title='Rate',
    legend_title='ID',
    template='plotly_white',
    hovermode='x'
)







# 2.0 DEEPECHO MODEL ----
model = PARModel(epochs=2048, cuda=True)


# it's not better with CUDA, because it's slower; just in case you ever want to try.
# model = PARModel(epochs=1024, cuda=True)


# Learn a model from the data
model.fit(
    data=calendar_filtered,
    entity_columns=['listing_id'],
    data_types=data_types,
    sequence_index='date'
)



# CREATE SYNTHETIC DATA ----
synthetic_result = model.sample(num_entities=10)
synthetic_result.glimpse()




# Sample details table for region
details = synthetic_result[['listing_id']].drop_duplicates()

# Generate the date range for each listing_id
date_range = pd.date_range(start='2024-07-08', end='2025-07-14')
listing_ids = synthetic_result['listing_id'].unique()

# Create a new DataFrame with listing_id and dates
date_df = pd.DataFrame({
    'listing_id': sorted(listing_ids.tolist() * len(date_range)),
    'date': list(date_range) * len(listing_ids)
})


# Add the date_yday to the date_df
# date_df['date_yday'] = date_df['date'].dt.dayofweek
date_df = (
    date_df.augment_timeseries_signature(date_column = "date")
)

# reduce data
date_df = date_df[['listing_id', 'date', 'date_index_num', 'date_year', 'date_half', 'date_quarter', 'date_month', 'date_yweek', 'date_wday', 'date_mday', 'date_yday']]




# Merge the date_df with the synthetic_result on listing_id and date_yday
merged_df = pd.merge(date_df, synthetic_result, on=['listing_id', 'date_index_num'], how='left')

# Group by and take the mean of duplicates
merged_df = merged_df.groupby(['listing_id', 'date', 'date_index_num'], as_index=False).agg({
    'rate': 'mean',
    'available': 'mean'
})

# Fill missing region values using a left join with the details table
merged_df = pd.merge(merged_df, details, on='listing_id', suffixes=('', '_detail'), how='left')

# Ensure there are 7 results for each listing_id
complete_df = date_df.merge(merged_df, on=['listing_id', 'date', 'date_index_num'], how='left')



complete_df['rate'] = complete_df.groupby('listing_id')['rate'].apply(lambda x: x.interpolate(method='linear')).values
complete_df['available'] = complete_df.groupby('listing_id')['available'].apply(lambda x: x.interpolate(method='linear')).values


# Fill any remaining NaN values with a reasonable value, e.g., 0
complete_df['rate'].fillna(0, inplace=True)
complete_df['available'].fillna(0, inplace=True)
complete_df['listing_id'] = complete_df['listing_id'].astype(str)


complete_df





# 4.0 BIND DATA & VISUALIZE ----
data_bound = pd.concat([calendar_filtered, complete_df], ignore_index=True)




custom_plot = go.Figure()

# Get unique symbols
symbols = data_bound['listing_id'].unique()

# Add a trace for each symbol
for listing_id in symbols:
    symbol_data = data_bound[data_bound['listing_id'] == listing_id]
    custom_plot.add_trace(go.Scatter(x=symbol_data['date'], y=symbol_data['rate'],
                             mode='lines', name=listing_id))


# Update the layout
custom_plot.update_layout(
    title='Daily Rate by Listing ID',
    xaxis_title='Date',
    yaxis_title='Rate',
    legend_title='ID',
    template='plotly_white',
    hovermode='x'
)

























