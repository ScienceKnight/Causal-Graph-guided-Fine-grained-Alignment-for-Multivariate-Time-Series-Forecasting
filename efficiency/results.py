import pandas as pd
import glob
import os

def aggregate_seq_length_results():
    all_files = glob.glob("sequence_length_efficiency_*.csv")
    combined = []
    for file in all_files:
        model_name = file.split