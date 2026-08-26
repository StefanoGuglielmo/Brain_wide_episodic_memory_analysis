# Brain_wide_episodic_memory_analysis

This repository contains the Python code used to reproduce the analyses and results presented in:

Guglielmo, S., Scantamburlo, M., Di Nardo, N., van den Oever, M., Mazziotti, R., Pizzorusso, T., & Origlia, N. (2026). Distributed neuronal ensembles support episodic-like memory retrieval. bioRxiv, 2026-06 (https://doi.org/10.64898/2026.06.15.727203)

The repository contains the Python notebooks and utility functions used for data analysis and figure generation. The processed datasets are not included in this GitHub repository and are provided separately through the associated Zenodo repository.

To reproduce the analyses, download both the GitHub repository and the associated Zenodo dataset, then run setup_data.py from the root directory of this repository. The script will verify and copy the required datasets from the downloaded Zenodo repository into the appropriate locations within the GitHub repository. Further instructions are provided in DATA/README.md.

Run the individual notebook in each folder, respecting the repository structure described below.

The analyses can be run using the environment specified in `environment.yml`. The separate environment specified in `environment_braian.yml` is required only for the Fig_3 notebook, which uses `braian` to generate the brain heatmaps.


## Repository structure

```text
.
├── DATA/
│   └── README.md
│
├── Fig_1/
│   └── Fig_1.ipynb
│
├── Fig_2/
│   └── Fig_2.ipynb
│
├── Fig_3/
│   └── Fig_3.ipynb
│
├── Fig_#/
│   └── Fig_#.ipynb
│
├── utils/
│   ├── io.py
│   ├── plot.py
│   ├── statistics.py
│   ├── network.py
│   └── ...
│
├── requirements.txt
├── requirements_braian.txt
├── environment.yml
├── environment_braian.yml
└── README.md
