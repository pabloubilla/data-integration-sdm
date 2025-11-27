import pandas as pd
import numpy as np
import os
from functools import reduce

### Data Pre Processing to map groups and to encode categorical variables
train_po_path = 'data/raw/NCEAS/Records/train_po'
output_path_po = 'data/processed/NCEAS/Records/train_po'
os.makedirs(output_path_po, exist_ok=True)
# get all csv files in train_po_path
train_po_files = [f for f in os.listdir(train_po_path) if f.endswith('.csv')]

group_mapping = {
    'ba': 'bat',
    'db': 'bird',
    'nb': 'bird',
    'sr': 'reptile',
    'rt': 'plant',
    'ou': 'plant',
    'ot': 'plant',
    'ru': 'plant'
}
group_mapping_keys = group_mapping.keys()

for file in train_po_files:
    df = pd.read_csv(os.path.join(train_po_path, file))
    # check if column 'group' exists
    if 'group' not in df.columns:
        continue


    # Map to same column, only replace if in mapping
    # df['group'] = df['group'].map(group_mapping).fillna(df['group'])

    groups = df['group'].unique()
    # if only one group, create a new file with the same name in output_path
    if len(groups) == 1:
        df.to_csv(os.path.join(output_path_po, file), index=False)
    else:
        # separate df by group and save each group in a new file
        for group in groups:
            df_group = df[df['group'] == group]
            output_file = f"{file.split('.')[0]}_{group}.csv"
            df_group.to_csv(os.path.join(output_path_po, output_file), index=False)


region = 'NSW'
keys = ['group', 'siteid', 'x', 'y']

# ---------- PA (merge horizontally on keys) ----------
test_pa_path = 'data/raw/NCEAS/Records/test_pa'
output_pa_path = 'data/processed/NCEAS/Records/test_pa'
os.makedirs(output_pa_path, exist_ok=True)

dfs_by_group = {}


for gr in group_mapping_keys:
    file_path = os.path.join(test_pa_path, f'{region}test_pa_{gr}.csv')
    if not os.path.exists(file_path):
        continue

    df = pd.read_csv(file_path)
    if 'group' not in df.columns:
        continue

    df['group'] = df['group'].map(group_mapping).fillna(df['group'])
    mapped_group = df['group'].iloc[0]

    dfs_by_group.setdefault(mapped_group, []).append(df)

for mapped_group, df_list in dfs_by_group.items():
    merged_df = reduce(lambda left, right: pd.merge(
        left, right, on=keys, how='outer'
    ), df_list)
    merged_df.fillna(0, inplace=True)
    out_file = os.path.join(output_pa_path, f'{region}test_pa_{mapped_group}.csv')
    merged_df = merged_df.sort_values("siteid").reset_index(drop=True)
    merged_df.to_csv(out_file, index=False)
    print(f"Saved PA -> {out_file}")

# ---------- ENV (concat vertically; infer group from filename code) ----------
train_env_path = 'data/raw/NCEAS/Records/test_env'
output_env_path = 'data/processed/NCEAS/Records/test_env'
os.makedirs(output_env_path, exist_ok=True)

dfs_by_group_env = {}

for gr in group_mapping_keys:
    file_path = os.path.join(train_env_path, f'{region}test_env_{gr}.csv')
    if not os.path.exists(file_path):
        continue

    df = pd.read_csv(file_path)
    # ENV files don't have 'group' column; infer from filename code and map
    mapped_group = group_mapping.get(gr, gr)
    df['group'] = mapped_group  # add group column so outputs carry the label

    dfs_by_group_env.setdefault(mapped_group, []).append(df)

for mapped_group, df_list in dfs_by_group_env.items():
    merged_df = pd.concat(df_list, ignore_index=True)
    # Drop the temporary group column before saving
    if 'group' in merged_df.columns:
        merged_df = merged_df.drop(columns=['group'])

    out_file = os.path.join(output_env_path, f'{region}test_env_{mapped_group}.csv')
    merged_df = merged_df.sort_values("siteid").reset_index(drop=True)
    merged_df.to_csv(out_file, index=False)
    print(f"Saved ENV -> {out_file}")