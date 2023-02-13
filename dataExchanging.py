import os
import pandas as pd
from zipfile import ZipFile
os.chdir('E:/Strategy/MachineLearning')
zip_file = ZipFile('data_formyFirstGithubRepo.zip')
dfs = {text_file.filename: pd.read_csv(zip_file.open(text_file.filename),encoding='unicode_escape',on_bad_lines='skip')
       for text_file in zip_file.infolist()
       if text_file.filename.endswith('.csv')}