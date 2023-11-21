# Overview

This repository contains code and data pertaining to [*Feature and order manipulations in a free recall task affect memory for current and future lists*](https://psyarxiv.com/erzfp) by Jeremy R. Manning, Emily C. Whitaker, Paxton C. Fitzpatrick, Madeline R. Lee, Allison M. Frantz, Bryan J. Bollinger, Darya Romanova, Campbell E. Field, and Andrew C. Heusser.

The repository is organized as follows:
```
root
├── code: analysis code used in the paper
│   ├── create_figures.ipynb: notebook for creating all figures
│   ├── dataloader.py: helper functions for downloading and and manipulating experimental data
│   ├── demographics.ipynb: notebook for compiling demographic information
│   ├── plot.py: helper functions for generating plots
│   ├── run_stats.ipynb: notebook for running statistical analyses
│   └── stimulus_table.ipynb: notebook for compiling information related to the stimuli
├── data: all data analyzed in the paper
│   ├── demographics.xlsx: demographic survey responses
│   └── eggs: stores the primary data files
│       ├── *.pkl files: saved computations
│       ├── *.egg files: downloaded when you run either the create_figures or run-stats notebooks; raw behavioral data
│       └── scratch: intermediate saved computations (*.pkl files)
├── paper: all files needed to generate a PDF of the paper and supplement
│   ├── compile.sh: run this to generate PDFs (requires a latex distribution with pdflatex and bibtex support)
│   ├── figures: pdf copies of all figures
├── requirements.txt: list of project dependencies
└── stimuli: stimuli used in the experiment
```

Our project uses [davos](https://github.com/ContextLab/davos) to improve shareability and compatability across systems.

# Setup instructions

Note: we have tested these instructions on MacOS and Ubuntu (Linux) systems.  We *think* they are likely to work on Windows systems too, but we haven't explicitly verified Windows compatability.

Note: if you would like to work with the raw (unprocessed) data, you'll need to install support for hdf5 on your system.  On MacOS this may be done using [Homebrew](https://brew.sh/):
  1. If you don't have Homebrew installed, run `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh` in Terminal to install it.
  2. To install hdf5 support, run `brew install hdf5`.
  
These steps are not necessary if you only want to reproduce the results and statistics reported in our paper, since those analyses depend on the preprocessed version of the dataset that is loaded by default when you run the notebooks.

We recommend running all of the analyses in a fresh Python 3.10 conda environment.  To set up your environment:
  1. Install [Anaconda](https://www.anaconda.com/)
  2. Clone this repository by running the following in a terminal: `git clone https://github.com/ContextLab/FRFR-analyses.git`  3. Create a new (empty) virtual environment by running the following (in the terminal): `conda create --name frfr python=3.10` (follow the prompts)
  3. Navigate (in terminal) to the activate the virtual environment (`conda activate frfr`)
  4. Install support for jupyter notebooks (`conda install -c anaconda ipykernel`) and then add the new kernel to your notebooks (`python -m ipykernel install --user --name=frfr`).  Follow any prompts that come up (accepting the default options should work).
  5. Install pytables (`conda install -c anaconda pytables`)
  6. Navigate to the `code` directory (`cd code`) in terminal
  7. Start a notebook server (`jupyter notebook`) and click on the notebook you want to run in the browser window that comes up.  `create_figures.ipynb` is a good place to start.  Selecting "Restart & Run All" from the "Kernel" menu will automatically
    - The first time you run the notebook, running the second cell will prompt you to restart the kernel and rerun the up to that point.  (This is needed in order to install the correct version of the `numpy` library automatically from within the notebook.)
    - After the notebook restarts and the first two cells run, select "Kernel" $\rightarrow$ "Restart & Run All" a second time to generate all of the figures in the paper and supplement.
  8. Make sure the notebook kernel is set to `frfr` (indicated in the top right).  If not, in the `Kernel` menu at the top of the notebook, select "Change kernel" and then "frfr".
  9. To stop the server, send the "kill" command in terminal (e.g., `ctrl` + `c` on a Mac or Linux system).
  10. To "exit" the virtual environment, type `conda deactivate`.

Notes:
- After setting up your environment for the first time, you can skip steps 1, 2, and 3 when you wish to re-enter the analysis environment in the future.
- After running the `create_figures.ipynb` notebook fully, the correct versions of all required packages for that notebook *and the other notebooks* will be automatically installed.  To run any other notebook:
  - Select the desired notebook from the Jupyter "Home Page" menu to open it in a new browser tab
  - Verify that the notebook is using the `frfr` kernel, using the above instructions to adjust the kernel if needed.
  - Select "Kernel" $\rightarrow$ "Restart & Run All" to execute all of the code in the notebook.

To remove the `frfr` environment from your system, run `conda remove --name frfr --all` in the terminal and follow the prompts.  (If you remove the `frfr` environment, you will need to repeat the initial setup steps if you want to re-run any of the code in the repository.)

Each notebook contains embedded documentation that describes what the various code blocks do.  Any figures you generate will end up in `paper/figures/source`.  Statistical results are printed directly from the notebooks when you run them.