
# LIBRARIES ----
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
import pytimetk as tk
import pandas as pd
import numpy as np

from sdv.sequential import PARSynthesizer
from sdv.metadata import SingleTableMetadata



# 0.0 IMPORT RAW DATA ----
combined_calendar = pd.read_csv("00_data/track_combined_calendar_svr.csv")
cluster_raw = pd.read_csv("00_data/kmeans_cluster_svr.csv")


# 1.0 DATA PREP
combined_calendar['rate'] = combined_calendar['rate'].astype(int)
combined_calendar['date'] = pd.to_datetime(combined_calendar['date'])



# Drop the unit_id column
svr_merged = pd.merge(combined_calendar, cluster_raw, left_on="listing_id", right_on="id", how = "left")
svr_merged = svr_merged.drop(columns=['available', 'id'])

# update with f string
svr_merged['listing_id'] = svr_merged['listing_id'].apply(lambda x: f'southern_vr_{x}')


svr_merged.glimpse()

svr_cluster_filter = svr_merged


# unique listings
unique_listings = svr_cluster_filter['listing_id'].unique()
first_n_listings = unique_listings[0:199]


# filter for the n listings, within 2024, and remove missing lat/lon
svr_prepared = svr_cluster_filter[svr_cluster_filter['listing_id'].isin(first_n_listings)]



current_year = pd.Timestamp.now().year
svr_prepared = svr_prepared[svr_prepared['date'].dt.year == current_year]


# FINALLY
svr_prepared.glimpse()


# Plotting with Plotly
fig = px.line(svr_prepared, x='date', y='rate', color='listing_id', title='Time Series of Rates by Listing ID')
fig.update_xaxes(title='Date')
fig.update_yaxes(title='Rate')
fig.show()



# create blank metadata & auto-detect
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(svr_prepared)


# manually update columns
metadata.update_columns_metadata(
    column_metadata={
        'listing_id': { 'sdtype': 'id' },
        'date': { 'sdtype': 'datetime' },
        'rate': { 'sdtype': 'numerical' },
        'cluster': { 'sdtype': 'categorical' },
        # 'available': { 'sdtype': 'boolean' },
        # 'bedrooms': { 'sdtype': 'categorical' },
        # 'bathrooms': { 'sdtype': 'categorical' },
        # 'sleeps': { 'sdtype': 'categorical' },
        # 'latitude': { 'sdtype': 'categorical' },
        # 'longitude': { 'sdtype': 'categorical' }
    }
)

# create relationshipo with lat/lon
# metadata.add_column_relationship(
#     relationship_type='gps',
#     column_names=['latitude', 'longitude']
# )


# set sequence stuff
metadata.set_sequence_key(column_name='listing_id')
metadata.set_sequence_index(column_name='date')


# VALIDATE & VISUALIZE
metadata.validate_data(data=svr_prepared)
metadata.visualize()


# save
# metadata.save_to_json(filepath='00_data/metadata_svr.json')





# 2.0 SYNTHESIZER ----

# Create the synthesizer
synthesizer = PARSynthesizer(
    metadata,
    epochs=250,
    context_columns=['cluster'],
    enforce_min_max_values=True,
    enforce_rounding=True,
    verbose=True
)


# Set constraints
# Rate constraint
rate_constraint = {
    'constraint_class': 'Positive',
    'constraint_parameters': {
        'column_name': 'rate',
        'strict_boundaries': True
    }
}


# bed/bath/sleeps constraints
# bed_bath_beyond_constraint = {
#     'constraint_class': 'Range',
#     'constraint_parameters': {
#         'low_column_name': 'bathrooms',
#         'middle_column_name': 'bedrooms',
#         'high_column_name': 'sleeps',
#         'strict_boundaries': True
#     }
# }



synthesizer.add_constraints(constraints=[
    rate_constraint,
    # bed_bath_beyond_constraint
])




# Train the synthesizer
synthesizer.fit(svr_prepared)



# Generate synthetic data
synthetic_data = synthesizer.sample(num_sequences=10)






# 4.0 BIND DATA & VISUALIZE ----
data_bound = pd.concat([svr_prepared, synthetic_data], ignore_index=True)


# Plotting with Plotly
fig = px.line(data_bound, x='date', y='rate', color='listing_id', title='Time Series of Rates by Listing ID')
fig.update_xaxes(title='Date')
fig.update_yaxes(title='Rate')
fig.show()



# 5.0 CUSTOM SYNTHESIZER

scenario_context = pd.DataFrame(data={
    'listing_id': ['listing_1a', 'listing_1b', 'listing_1c', 'listing_2a', 'listing_2b'],
    'cluster': ['cluster_1']*3 + ['cluster_2']*2
})

scenario_context


synthesizer.sample_sequential_columns(
    context_columns=scenario_context
)







