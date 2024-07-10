
# LIBRARIES ----
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
import pytimetk as tk
import pandas as pd
import numpy as np

from deepecho import PARModel
from deepecho.demo import load_demo




# 1.0 DATA IMPORT ----

data = load_demo()

# Define data types for all the columns
data_types = {
    'region': 'categorical',
    'day_of_week': 'categorical',
    'total_sales': 'continuous',
    'nb_customers': 'count',
}


# VISUALIZE ----
data['date'] = pd.to_datetime(data['date'])
data['store_id'] = data['store_id'].astype(str)
data.glimpse()


custom_plot = go.Figure()

# Get unique symbols
symbols = data['store_id'].unique()

# Add a trace for each symbol
for store_id in symbols:
    symbol_data = data[data['store_id'] == store_id]
    custom_plot.add_trace(go.Scatter(x=symbol_data['date'], y=symbol_data['total_sales'],
                             mode='lines', name=store_id))


# Update the layout
custom_plot.update_layout(
    title='Sales by Store ID',
    xaxis_title='Date',
    yaxis_title='Total Sales',
    legend_title='Symbol',
    template='plotly_white',
    hovermode='x'
)




# 2.0 CREATE THE MODEL ----
model = PARModel(epochs=2048, cuda=False)


# it's not better with CUDA, because it's slower; just in case you ever want to try.
# model = PARModel(epochs=1024, cuda=True)


# Learn a model from the data
model.fit(
    data=data,
    entity_columns=['store_id'],
    context_columns=['region'],
    data_types=data_types,
    sequence_index='date'
)




# 3.0 CREATE SYNTHETIC DATA ----
# Sample new data
synthetic_result = model.sample(num_entities=50)
synthetic_result.glimpse()


# Sample details table for region
details = synthetic_result[['store_id', 'region']].drop_duplicates()

# Generate the date range for each store_id
date_range = pd.date_range(start='2020-06-01', end='2020-06-07')
store_ids = synthetic_result['store_id'].unique()

# Create a new DataFrame with store_id and dates
date_df = pd.DataFrame({
    'store_id': sorted(store_ids.tolist() * len(date_range)),
    'date': list(date_range) * len(store_ids)
})

# Add the day_of_week to the date_df
date_df['day_of_week'] = date_df['date'].dt.dayofweek

# Merge the date_df with the synthetic_result on store_id and day_of_week
merged_df = pd.merge(date_df, synthetic_result, on=['store_id', 'day_of_week'], how='left')

# Group by and take the mean of duplicates
merged_df = merged_df.groupby(['store_id', 'date', 'day_of_week', 'region'], as_index=False).agg({
    'total_sales': 'mean',
    'nb_customers': 'mean'
})

# Fill missing region values using a left join with the details table
merged_df = pd.merge(merged_df, details, on='store_id', suffixes=('', '_detail'), how='left')
merged_df['region'] = merged_df['region'].combine_first(merged_df['region_detail'])
merged_df.drop(columns=['region_detail'], inplace=True)

# Ensure there are 7 results for each store_id
complete_df = date_df.merge(merged_df, on=['store_id', 'date', 'day_of_week'], how='left')

# impute missing values for total_sales and nb_customers
complete_df['total_sales'] = complete_df.groupby('store_id')['total_sales'].apply(lambda x: x.interpolate(method='linear')).values
complete_df['nb_customers'] = complete_df.groupby('store_id')['nb_customers'].apply(lambda x: x.interpolate(method='linear')).values


# Fill any remaining NaN values with a reasonable value, e.g., 0
complete_df['total_sales'].fillna(0, inplace=True)
complete_df['nb_customers'].fillna(0, inplace=True)
complete_df['store_id'] = complete_df['store_id'].astype(str)


complete_df



# 4.0 BIND DATA & VISUALIZE ----
data_bound = pd.concat([data, complete_df], ignore_index=True)




custom_plot = go.Figure()

# Get unique symbols
symbols = data_bound['store_id'].unique()

# Add a trace for each symbol
for store_id in symbols:
    symbol_data = data_bound[data_bound['store_id'] == store_id]
    custom_plot.add_trace(go.Scatter(x=symbol_data['date'], y=symbol_data['total_sales'],
                             mode='lines', name=store_id))


# Update the layout
custom_plot.update_layout(
    title='Sales by Store ID',
    xaxis_title='Date',
    yaxis_title='Total Sales',
    legend_title='Symbol',
    template='plotly_white',
    hovermode='x'
)




















