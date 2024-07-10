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
combined_listings = pd.read_csv("00_data/combined_listings_svr.csv")
combined_calendar = pd.read_csv("00_data/track_combined_calendar_svr.csv")


# 1.0 DATA PREP
svr_merged = pd.merge(combined_calendar, combined_listings, left_on="listing_id", right_on="unit_id", how = "left")

svr_merged['rate'] = svr_merged['rate'].astype(int)
svr_merged['date'] = pd.to_datetime(svr_merged['date'])

# fill any na's with 0
svr_merged['bedrooms'] = svr_merged['bedrooms'].fillna(0).astype(int)
svr_merged['bedrooms'] = svr_merged['bedrooms'].astype(int)

svr_merged['bathrooms'] = svr_merged['bathrooms'].astype(int)
svr_merged['sleeps'] = svr_merged['sleeps'].astype(int)

svr_merged['available'] = svr_merged['available'].astype(bool)


# Drop the unit_id column
svr_merged = svr_merged.drop(columns=['unit_id'])

# update with f string
svr_merged['listing_id'] = svr_merged['listing_id'].apply(lambda x: f'southern_vr_{x}')


svr_merged.glimpse()


# unique listings
unique_listings = svr_merged['listing_id'].unique()
first_n_listings = unique_listings[0:49]


# filter for the n listings, within 2024, and remove missing lat/lon
svr_prepared = svr_merged[svr_merged['listing_id'].isin(first_n_listings)]

svr_prepared = svr_prepared.dropna(subset=['latitude'])

# current_year = pd.Timestamp.now().year
# svr_prepared = svr_prepared[svr_prepared['date'].dt.year == current_year]


# FINALLY
svr_prepared.glimpse()





# create blank metadata & auto-detect
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(svr_prepared)


# manually update columns
metadata.update_columns_metadata(
    column_metadata={
        'listing_id': { 'sdtype': 'id' },
        'rate': { 'sdtype': 'numerical', 'computer_representation': 'Int64' },
        'available': { 'sdtype': 'boolean' },
        'bedrooms': { 'sdtype': 'categorical' },
        'bathrooms': { 'sdtype': 'categorical' },
        'sleeps': { 'sdtype': 'categorical' },
        'latitude': { 'sdtype': 'categorical' },
        'longitude': { 'sdtype': 'categorical' }
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
    enforce_min_max_values=False,
    enforce_rounding=True,
    context_columns=['latitude', 'longitude', 'bedrooms', 'bathrooms', 'sleeps'],
    cuda = True
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



custom_plot = go.Figure()

# Get unique symbols
listing_symbols = []
listing_symbols = data_bound['listing_id'].unique()

# Add a trace for each symbol
for listing_id in listing_symbols:
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













