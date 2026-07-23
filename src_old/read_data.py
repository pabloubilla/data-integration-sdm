import pandas as pd
import numpy as np
import torch.nn as nn
from src_old.models import DeepMaxEntModel, DeepMaxEntLoss

add_PO_var = True

df_NZ_po = pd.read_csv('data/Records/train_po/AWTtrain_po.csv')
df_NZ_po_bird = df_NZ_po[df_NZ_po['group'] == 'bird']

df_NZ_pa_bird = pd.read_csv('data/Records/test_pa/AWTtest_pa_bird.csv')
df_NZ_env_bird = pd.read_csv('data/Records/test_env/AWTtest_env_bird.csv')

# pivot df_NZ_po_bird to have one column per species with 1 for presence and 0 for absence
unique_species = df_NZ_po_bird['spid'].unique()
Y = df_NZ_po_bird.pivot_table(index=['x','y'], columns='spid', aggfunc='size', fill_value=0).reset_index()
### one point to consider is if different observations for the same site should have the same covariates
covariates = ["bc01","bc04","bc05","bc06","bc12","bc15","bc17","bc20","bc31","bc33"] # seem to be the covariates
X = df_NZ_po_bird.groupby(['x','y'])[covariates].mean().reset_index()
if add_PO_var: X['PO'] = 1 # add a column for presence only

Y_test = df_NZ_pa_bird[unique_species]
X_test = df_NZ_env_bird[covariates]
if add_PO_var: X_test['PO'] = 0 # add a column for presence only
# separate X_test and Y_test into two datasets (0.5, 0.5) 
from sklearn.model_selection import train_test_split
X_test, X_extra, Y_test, Y_extra = train_test_split(X_test, Y_test, test_size=0.5, random_state=42)
# add the extra data to the training data
X = pd.concat([X, X_extra])
Y = pd.concat([Y, Y_extra])

# add PO to covariates
if add_PO_var: covariates.append('PO')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X[covariates])
X[covariates] = scaler.transform(X[covariates])
X_test[covariates] = scaler.transform(X_test[covariates])

deep_maxent = DeepMaxEntModel(input_size=len(covariates), 
                               hidden_size=256, output_size=len(unique_species), hidden_nbr=3)

# train the model
criterion = DeepMaxEntLoss()
import torch
optimizer = torch.optim.Adam(deep_maxent.parameters(), lr=0.0001)

X_tensor = torch.tensor(X[covariates].values, dtype=torch.float32)
Y_tensor = torch.tensor(Y[unique_species].values, dtype=torch.float32)

n_epochs = 10000
for epoch in range(n_epochs):
    optimizer.zero_grad()
    outputs = deep_maxent(X_tensor)
    loss = criterion(outputs, Y_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')


# evaluate the model
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test[covariates].values, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test[unique_species].values, dtype=torch.float32)
    test_outputs = deep_maxent(X_test_tensor)
    test_loss = criterion(test_outputs, Y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

# evaluate the model using AUC for each species
from sklearn.metrics import roc_auc_score
test_outputs_np = test_outputs.numpy()
auc_scores = {}
for i, species in enumerate(unique_species):
    auc = roc_auc_score(Y_test[species], test_outputs_np[:, i])
    print(f'Species: {species}, AUC: {auc:.4f}')
    auc_scores[species] = auc

# print average
average_auc = np.mean(list(auc_scores.values()))
print(average_auc)
