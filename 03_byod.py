
# LIBRARIES ----
import pandas as pd
import numpy as np
import janitor as jn

from sdv.datasets.local import load_csvs
from sdv.datasets.demo import download_demo


# 1.0 RAW DATA IMPORT ----
# data folder
FOLDER_NAME = 'content/'

try:
  data = load_csvs(folder_name='/content/')
except ValueError:
  print('You have not uploaded any csv files. Using some demo data instead.')
  data, _ = download_demo(
    modality='multi_table',
    dataset_name='fake_hotels'
  )





















