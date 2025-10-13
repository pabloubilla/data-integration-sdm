import pandas as pd
import numpy as np
import os


train_po_path = 'data/raw/Records/train_po'
output_path = 'data/processed/Records/train_po'
os.makedirs(output_path, exist_ok=True)
# get all csv files in train_po_path
train_po_files = [f for f in os.listdir(train_po_path) if f.endswith('.csv')]

for file in train_po_files:
    df = pd.read_csv(os.path.join(train_po_path, file))
    # check if column 'group' exists
    if 'group' not in df.columns:
        continue
    groups = df['group'].unique()
    # if only one group, create a new file with the same name in output_path
    if len(groups) == 1:
        df.to_csv(os.path.join(output_path, file), index=False)
    else:
        # separate df by group and save each group in a new file
        for group in groups:
            df_group = df[df['group'] == group]
            output_file = f"{file.split('.')[0]}_{group}.csv"
            df_group.to_csv(os.path.join(output_path, output_file), index=False)