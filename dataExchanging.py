import pandas as pd
import os
import zipfile
os.chdir('E:\Strategy\MachineLearning')
with zipfile.ZipFile('data_formyFirstGithubRepo.zip','r') as zip_ref:
    for filename in zip_ref.namelist():
        if filename.endswith('.csv'):
            df = pd.read_csv(zip_ref.open(filename))
