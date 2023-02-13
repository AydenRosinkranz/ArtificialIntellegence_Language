import os
import pandas as pd
from zipfile import ZipFile
import re

os.chdir('E:/Strategy/MachineLearning')
zip_file = ZipFile('data_formyFirstGithubRepo.zip')
dfs = {text_file.filename: pd.read_csv(zip_file.open(text_file.filename),encoding='unicode_escape',on_bad_lines='skip')
       for text_file in zip_file.infolist()
       if text_file.filename.endswith('.csv')}
# def writedf(dfcontent: pandas.core.frame.DataFrame) -> pandas.core.frame.DataFrame:
#        # Remind myself that each index is writen in str type
#        dfcontent = np.array(dfcontent)
#        column_names = dfcontent[:,0]
#        data = dfcontent[:,1:]
#        var_value = pd.DataFrame(data.T,columns=column_names)
#        return var_value
# for var_name,dfcontent in dfs.items():
#        var_value = writedf(dfcontent)
#        exec(var_name+f"= {var_value}")
for var_name,dfcontent in dfs.items():
       var_name = re.search('/([^.]*)\.', var_name).group(1)
       a = dfcontent
       exec(f"{var_name} = a")

