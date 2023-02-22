import os
import glob
os.environ['CUDA_VISIBLE_DEVICES']=''
# Get a list of all Python files in the folder
files = glob.glob('E:\Strategy\MachineLearning\ArtificialIntelligence_Language\*.py')
# Execute each file
for file in files:
    exec(open(file).read())