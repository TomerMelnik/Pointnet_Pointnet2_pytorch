
import os
import requests
import zipfile
import shutil
import streamlit as st

# Function to download and extract data
def download_and_extract_data(url, extract_to='./data'):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    zip_path = os.path.join(extract_to, 'shapenet.zip')
    
    # Download the file
    if not os.path.exists(zip_path):
        st.write(f"Downloading data from {url}...")
        response = requests.get(url, stream=True, verify=False)
        with open(zip_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        st.write("Download complete.")

    # Extract the file
    extracted_data_path = os.path.join(extract_to, 'shapenetcore_partanno_segmentation_benchmark_v0_normal')
    if not os.path.exists(extracted_data_path):
        st.write("Extracting data...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        st.write("Extraction complete.")
    else:
        st.write("Data already extracted.")

# URL to download the data from
DATA_URL = 'https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip'  # Replace with actual URL

# Download and extract data if not already done
download_and_extract_data(url=DATA_URL)
