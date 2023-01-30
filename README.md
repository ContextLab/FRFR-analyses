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

We recommend running all of the analyses in a fresh Python 3.10 conda environment.  To set up your environment:
  1. Install [Anaconda](https://www.anaconda.com/)
  2. Clone this repository by running the following in a terminal: `git clone https://github.com/ContextLab/FRFR-analyses.git`
  3. Create a new (empty) virtual environment by running the following (in the terminal): `conda create --name frfr python=3.10` (follow the onscreen prompts)
  4. Navigate (in terminal) to the activate the virtual environment (`conda activate frfr`)
  5. Install project requirements: `pip install -r requirements.txt`
  6. Navigate to the `code` directory (`cd code`) in terminal
  7. Start a notebook server (`jupyter notebook`) and click on the notebook you want to run in the browser window that comes up.  `create_figures.ipynb` is a good place to start.  To stop the server, send the "kill" command in terminal (e.g., `ctrl` + `c` on a Mac or Linux system).
  8. To "exit" the virtual environment, type `conda deactivate`.

After setting up your environment for the first time, you can skip steps 1, 2, 3, and 5 when you wish to re-enter the analysis environment in the future.

Each notebook contains embedded documentation that describes what the various code blocks do.  Any figures you generate will end up in `paper/figures/source`.  Statistical results are printed directly from the notebooks when you run them.

