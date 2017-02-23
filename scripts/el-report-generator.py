import nbformat as nbf
from subprocess import call
import os
import seaborn as sns

nb = nbf.v4.new_notebook()
cwd = os.getcwd()

nb['cells'] = []

# first cell
text = "## Import Libraries"
code = "import matplotlib as mpl, \
        matplotlib.pyplot as plt, \
        seaborn as sns; \
        get_ipython().magic('matplotlib inline')"

nb['cells'].append(nbf.v4.new_markdown_cell(text))
nb['cells'].append(nbf.v4.new_code_cell(code))

#second cell
text = "## Plot something"
code = "sns.tsplot([1,2,3])"

nb['cells'].append(nbf.v4.new_markdown_cell(text))
nb['cells'].append(nbf.v4.new_code_cell(code))

fname = 'test.ipynb'

with open(fname, 'w') as f:
    nbf.write(nb, f)

call('jupyter nbconvert --to html --execute /Users/andyheusser/Documents/github/FRFR-analyses/test.ipynb', shell=True)
