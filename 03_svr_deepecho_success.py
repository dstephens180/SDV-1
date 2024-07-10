
# LIBRARIES ----
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
import pytimetk as tk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from deepecho import PARModel

# this step is for metadata.visualize for GraphViz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'





# 0.0 IMPORT RAW DATA ----
combined_calendar = pd.read_csv("00_data/track_combined_calendar_svr.csv")
cluster_raw = pd.read_csv("00_data/kmeans_cluster_svr.csv")





# 1.0 DATA PREP
combined_calendar['rate'] = combined_calendar['rate'].astype(int)



# Drop the unit_id column
svr_merged = pd.merge(combined_calendar, cluster_raw, left_on = "listing_id", right_on = "id", how = "left")
svr_merged = svr_merged.drop(columns=['available'])

# update with f string
svr_merged['listing_id'] = svr_merged.apply(lambda row: f'{row["cluster"]}_{row["listing_id"]}', axis = 1)



svr_merged['sequential_count'] = svr_merged.groupby('listing_id').cumcount()

svr_merged.glimpse()


svr_timestamp = svr_merged[['listing_id', 'date', 'rate', 'sequential_count', 'cluster']]


# filter n samples from each cluster
random_listings = svr_timestamp.groupby('cluster').apply(lambda x: x.sample(n=5)).reset_index(drop=True)

# Extract the listing_id column
listing_ids = random_listings['listing_id'] \
    .tolist()



# filter only cluster
# filter_cluster = ['Bliss', 'Meadow', 
#                   'Harmony', 'Crescent', 
#                   'Radiant', 'Serenity', 
#                   'Sunset', 'Whisper', 'Willow'
#                   ]


svr_cluster_filter = svr_timestamp[svr_timestamp['listing_id'].isin(listing_ids)]






# unique listings
# unique_listings = svr_cluster_filter['listing_id'].unique()
# first_n_listings = unique_listings[0:19]

# only dates within 2024
# current_year = pd.Timestamp.now().year
# svr_cluster_filter['datetime'] = pd.to_datetime(svr_cluster_filter['date'])
# svr_cluster_filter = svr_cluster_filter[svr_cluster_filter['datetime'].dt.year == current_year]


svr_prepared = svr_cluster_filter[['listing_id', 'date', 'rate', 'sequential_count', 'cluster']]



# FINALLY
svr_prepared.glimpse()



# Define data types for all the columns
data_types = {
    'cluster': 'categorical',
    'sequential_count': 'categorical',
    'rate': 'continuous',
}


# Plotting with Plotly
fig = px.line(svr_prepared, x='date', y='rate', color='listing_id', title='Time Series of Rates by Listing ID')
fig.update_xaxes(title='Date')
fig.update_yaxes(title='Rate')
fig.show()






# 2.0 DEEPECHO MODEL ----
model = PARModel(epochs=2048, cuda=True)



# Learn a model from the data
model.fit(
    data=svr_prepared,
    entity_columns=['listing_id'],
    data_types=data_types,
    sequence_index='date'
)



# CREATE SYNTHETIC DATA ----
min_date = pd.to_datetime(min(svr_prepared['date']))
max_date = pd.to_datetime(max(svr_prepared['date']))
diff_date = (max_date - min_date).days

synthetic_result = model.sample(
    num_entities=20,
    sequence_length = diff_date
)

synthetic_result.glimpse()


# Sample details table for region
details = synthetic_result[['listing_id', 'cluster']].drop_duplicates()


# Generate the date range for each listing_id
date_range = pd.date_range(start=min_date, end=max_date)
listing_ids = synthetic_result['listing_id'].unique()

# Create a new DataFrame with listing_id and dates
date_df = pd.DataFrame({
    'listing_id': sorted(listing_ids.tolist() * len(date_range)),
    'date': list(date_range) * len(listing_ids)
})


# Add the date_yday to the date_df
date_df['sequential_count'] = date_df.groupby('listing_id').cumcount()


# reduce data
date_df = date_df[['listing_id', 'date', 'sequential_count']]




# Merge the date_df with the synthetic_result on listing_id and date_yday
merged_df = pd.merge(synthetic_result, date_df, on=['listing_id', 'sequential_count'], how='left')

# Group by and take the mean of duplicates
merged_df = merged_df.groupby(['listing_id', 'date', 'sequential_count', 'cluster'], as_index=False).agg({
    'rate': 'mean'
})

# Fill missing cluster values using a left join with the details table
# merged_df = pd.merge(merged_df, details, on='listing_id', suffixes=('', '_detail'), how='left')
# merged_df['cluster'] = merged_df['cluster'].combine_first(merged_df['cluster_detail'])
# merged_df.drop(columns=['cluster_detail'], inplace=True)

complete_df = merged_df

complete_df \
    .groupby('listing_id') \
    .pad_by_time(
        date_column = 'date',
        freq        = 'D',
        start_date  = min_date,
        end_date    = max_date
    )


# Function to perform cyclic linear interpolation
# def cyclic_interpolate(group):
#     # Ensure the index represents the day of the year
#     group.index = group.index % 365
#     # Sort by the index (day of the year)
#     group = group.sort_index()
#     # Perform linear interpolation
#     interpolated = group.interpolate(method='linear')
#     # If any values are still NaN, fill them by wrapping around
#     if interpolated.isna().sum() > 0:
#         interpolated = interpolated.interpolate(method='linear', limit_direction='both')
#     return interpolated

# # Assume complete_df is your DataFrame and has a datetime index or a day_of_year index
# # If it's a datetime index, convert it to day_of_year
# if pd.api.types.is_datetime64_any_dtype(complete_df.index):
#     complete_df['sequential_count'] = complete_df.index.dayofyear
#     complete_df.set_index('sequential_count', inplace=True)

# # Apply the cyclic interpolation function
# complete_df['rate'] = complete_df.groupby('listing_id')['rate'].apply(cyclic_interpolate).values

# # Reset the index if needed
# if 'sequential_count`' in complete_df.columns:
#     complete_df.reset_index(drop=True, inplace=True)



# # Fill any remaining NaN values with a reasonable value, e.g., 0
# complete_df['rate'].fillna(0, inplace=True)
# complete_df['listing_id'] = complete_df['listing_id'].astype(str)


# update with f string
complete_df['listing_id'] = complete_df.apply(lambda row: f'{row["cluster"]}_synth_{row["listing_id"]}', axis = 1)



complete_df





# 4.0 BIND DATA & VISUALIZE ----
data_bound = pd.concat([svr_prepared, complete_df], ignore_index=True)




# plot
fig = px.line(data_bound, x='date', y='rate', color='listing_id', title='Time Series of Rates by Listing ID')
fig.update_xaxes(title='Date')
fig.update_yaxes(title='Rate')
fig.show()










# 5.0 CUSTOM SYNTHESIZER ----
scenario_context = pd.DataFrame(data={
    'cluster': ['Bliss']*10
})

scenario_context



custom_result = model.sample(num_entities=1, context = scenario_context, sequence_length = diff_date)


# filter only cluster
filter_cluster = ['Bliss']
custom_result_filter = custom_result[custom_result['cluster'].isin(filter_cluster)] \
    .sort_values(by = "sequential_count") \
    .groupby(['listing_id', 'sequential_count'], as_index=False) \
    .agg({'rate': 'mean'})




# plot
fig = px.line(custom_result_filter, x='sequential_count', y='rate', color='listing_id', title='Time Series of Rates by Listing ID')
fig.update_xaxes(title='Date')
fig.update_yaxes(title='Rate')
fig.show()






