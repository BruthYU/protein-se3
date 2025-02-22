# -*- coding: utf-8 -*-
"""
download_pdb.py
~~~~~~~~~~~~~~~

Use PDB ID to download PDB files

Authored by Chris Swain (http://www.macinchem.org)
Released into the public domain

"""

import csv
import os
import sys
import pandas as pd
from urllib.request import urlretrieve
from tqdm import tqdm
os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'

# pdb_codes_file = '../lightning/resource/frameflow/targets'
# print(os.listdir(pdb_codes_file))
#
# # File containing comma-separated list of the desired PDB IDs
# pdb_codes_file = '../lightning/resource/frameflow/benchmark.csv'

# pdb_df = pd.read_csv(pdb_codes_file)
# pdb_codes = list(pdb_df['target'])


# Whether to download gzip compressed files
compressed = False
# Folder to download files to
download_folder = './raw/pdb'
# Alternatively, hard code the PDB IDs:
pdb_codes = ['1LS6', '1Z28', '2D06', '3QVU', '3QVV', '3U3J', '3U3K']

print(pdb_codes)

# Ensure download folder exists
try:
    os.makedirs(download_folder)
except OSError as e:
    # Ignore OSError raised if it already exists
    pass

for pdb_code in tqdm(pdb_codes):
    # Add .pdb extension and remove ':1' suffix if entities
    filename = '%s.pdb' % pdb_code[:4].upper()
    # Add .gz extension if compressed
    if compressed:
        filename = '%s.gz' % filename
    url = 'https://files.rcsb.org/download/%s' % filename
    destination_file = os.path.join(download_folder, filename)
    if os.path.exists(destination_file):
        continue
    # Download the file
    urlretrieve(url, destination_file)
