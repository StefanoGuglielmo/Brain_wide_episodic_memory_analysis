The datasets required to run the analyses presented in this repository are provided separately through the associated Zenodo repository.

The datasets are not included in the GitHub repository. To reproduce the analyses:

Download the complete dataset from the associated Zenodo repository and extract it locally.
Download or clone this GitHub repository.
From the root directory of the GitHub repository, run:
python setup_data.py
When prompted, provide the path to the folder containing the datasets downloaded from Zenodo.

The script will verify that all required datasets are present and copy them into the appropriate locations within the GitHub repository, including the DATA/ directory and the directories associated with individual figures.

The script does not modify the files downloaded from Zenodo.

Once the data have been populated, the analysis notebooks can be run according to the instructions provided in the main README.md.

The Zenodo repository contains the archived datasets associated with this version of the code.