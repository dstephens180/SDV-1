
# LIBRARIES ----
import pandas as pd
import numpy as np
import plotly.io as pio
import plotly.express as px

from sdv.datasets.demo import download_demo



# 0.0 DATA IMPORT ----

real_data, metadata = download_demo(
    modality='single_table',
    dataset_name='fake_hotel_guests')


real_data
metadata


# 1.0 SYNTHETIC DATA ----
from sdv.single_table import GaussianCopulaSynthesizer


# use gaussian copula as the SDV synthesizer, which creates synthetic data.
# It learns patterns from real data and replicates them to create synthetic data.
synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(data=real_data)


# generate 500 rows of synthetic data
synthetic_data = synthesizer.sample(num_rows=500)
synthetic_data



# 2.0 EVALUATE SYNTHETIC DATA ----
from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import run_diagnostic
from sdv.evaluation.single_table import get_column_plot

quality_report = evaluate_quality(
    real_data,
    synthetic_data,
    metadata)

diagnostic = run_diagnostic(
    real_data,
    synthetic_data,
    metadata)

fig = get_column_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    column_name='room_rate',
    metadata=metadata,
    plot_type="distplot"
)


# visualize; must run pio.renders before fig.show()
pio.renderers.default = "vscode"
fig.show()

